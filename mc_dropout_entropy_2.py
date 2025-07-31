import torch
from torch import Tensor, nn
from torch.nn.modules.dropout import _DropoutNd
from typing import List, Dict, Optional
from einops import rearrange, repeat
import warnings


class MCDropout(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_estimators: int,
        last_layer: bool,
        on_batch: bool,
    ) -> None:
        """MC Dropout wrapper for a model containing nn.Dropout modules.

        Args:
            model (nn.Module): model to wrap
            num_estimators (int): number of estimators to use during the
                evaluation
            last_layer (bool): whether to apply dropout to the last layer only.
            on_batch (bool): Perform the MC-Dropout on the batch-size.
                Otherwise in a for loop. Useful when constrained in memory.

        Warning:
            This module will work only if you apply dropout through modules
            declared in the constructor (__init__).

        Warning:
            The `last-layer` option disables the lastly initialized dropout
            during evaluation: make sure that the last dropout is either
            functional or a module of its own.
        """
        super().__init__()
        filtered_modules = list(
            filter(
                lambda m: isinstance(m, _DropoutNd),
                model.modules(),
            )
        )
        if last_layer:
            filtered_modules = filtered_modules[-1:]

        _dropout_checks(filtered_modules, num_estimators)
        self.last_layer = last_layer
        self.on_batch = on_batch
        self.core_model = model
        self.num_estimators = num_estimators
        self.filtered_modules = filtered_modules

    def train(self, mode: bool = True) -> nn.Module:
        """Override the default train method to set the training mode of
        each submodule to be the same as the module itself except for the
        selected dropout modules.

        Args:
            mode (bool, optional): whether to set the module to training
                mode. Defaults to True.
        """
        if not isinstance(mode, bool):
            raise TypeError("Training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        for module in self.filtered_modules:
            module.train()
        return self

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Forward pass of the model.

        During training, the forward pass is the same as of the core model.
        During evaluation, the forward pass is repeated `num_estimators` times
        either on the batch size or in a for loop depending on
        :attr:`last_layer`.

        Args:
            x (Tensor): input tensor of shape (B, ...)

        Returns:
            Tensor: output tensor of shape (:attr:`num_estimators` * B, ...)
        """
        if self.training:
            return self.core_model(x)

        if self.on_batch:
            b = x.shape[0]
            x = repeat(x, "b c h w -> b n c h w", n=self.num_estimators)

            list_uncertainty = []
            list_logists = []

            for i in range(b):
                x_ = x[i]
                logists_all = self.core_model(x_)
                logists = logists_all.mean(dim=0)
                list_logists.append(logists)

                probs_all = torch.softmax(logists_all, dim=-1)
                dict_uncertainty = calculate_classification_uncertainties(probs_all)
                list_uncertainty.append(dict_uncertainty)

            logists_final = torch.stack(list_logists, dim=0)

            return logists_final, list_uncertainty

def mc_dropout(
    model: nn.Module,
    num_estimators: int,
    last_layer: bool = False,
    on_batch: bool = True,
) -> MCDropout:
    """MC Dropout wrapper for a model.

    Args:
        model (nn.Module): model to wrap
        num_estimators (int): number of estimators to use
        last_layer (bool, optional): whether to apply dropout to the last
            layer only. Defaults to False.
        on_batch (bool): Increase the batch_size to perform MC-Dropout.
            Otherwise in a for loop to reduce memory footprint. Defaults
            to true.
    """
    return MCDropout(
        model=model,
        num_estimators=num_estimators,
        last_layer=last_layer,
        on_batch=on_batch,
    )


def _dropout_checks(filtered_modules: List[nn.Module], num_estimators: int) -> None:
    if not filtered_modules:
        raise ValueError(
            "No dropout module found in the model. "
            "Please use `nn.Dropout`-like modules to apply dropout."
        )
    # Check that at least one module has > 0.0 dropout rate
    if not any(mod.p > 0.0 for mod in filtered_modules):
        raise ValueError("At least one dropout module must have a dropout rate > 0.0.")
    if num_estimators <= 0:
        raise ValueError(
            "`num_estimators` must be strictly positive to use MC Dropout."
        )


def calculate_classification_uncertainties(predictions):
    """
    计算分类任务中的四种不确定性指标

    参数:
    predictions - torch.Tensor, 形状为 (n_estimates, n_classes)
                 包含多个Dropout采样得到的类别概率分布

    返回:
    dict - 包含以下四种不确定性指标的字典:
        'predictive_entropy': 预测熵
        'mutual_information': 互信息
        'pred_class_variance': 预测类别的概率方差
        'pred_class_coef_of_variation': 预测类别的变异系数
    """
    # 确保输入是有效的概率分布
    assert torch.all(predictions >= 0), "概率值不能为负数"
    assert torch.allclose(predictions.sum(dim=-1), torch.tensor(1.0)), (
        "每行应该是概率分布"
    )

    results = {}

    # 计算平均预测概率分布
    mean_probs = predictions.mean(dim=0)

    # 1. 预测熵 (Predictive Entropy)
    # 计算平均分布的熵: H[E[p(y|x)]]
    log_probs = torch.log(mean_probs + 1e-10)  # 防止log(0)
    predictive_entropy = -torch.sum(mean_probs * log_probs)
    results["predictive_entropy"] = predictive_entropy.item()

    # 2. 互信息 (Mutual Information)
    # MI = H[E[p(y|x)]] - E[H[p(y|x)]]

    # 首先计算每个估计的熵
    per_estimate_entropies = -torch.sum(
        predictions * torch.log(predictions + 1e-10), dim=1
    )

    # 计算期望熵 E[H[p(y|x)]]
    expected_entropy = per_estimate_entropies.mean()

    # 计算互信息
    mutual_information = predictive_entropy - expected_entropy
    results["mutual_information"] = mutual_information.item()

    # 3. 预测类别的概率方差
    # 确定预测类别 (平均概率最高的类别)
    pred_class = mean_probs.argmax()

    # 获取该类别在所有估计中的概率值
    pred_class_probs = predictions[:, pred_class]

    # 计算方差
    pred_class_variance = pred_class_probs.var(unbiased=True)
    results["pred_class_variance"] = pred_class_variance.item()

    # 4. 预测类别的变异系数 (Coefficient of Variation)
    pred_class_mean = pred_class_probs.mean()
    pred_class_std = pred_class_probs.std(unbiased=True)
    coef_of_variation = pred_class_std / (pred_class_mean + 1e-10)
    results["pred_class_coef_of_variation"] = coef_of_variation.item()

    # # 添加辅助信息
    # results["_predicted_class"] = pred_class.item()
    # results["_predicted_prob"] = pred_class_mean.item()

    return results


# def calculate_classification_uncertainties(
#     predictions: torch.Tensor, true_label: Optional[int] = None, epsilon: float = 1e-10
# ) -> Dict[str, float]:
#     """
#     计算分类任务中的各种不确定性指标

#     参数:
#     predictions: torch.Tensor, 形状为 (n_estimates, n_classes)
#                 包含多个MC Dropout采样得到的类别概率分布
#     true_label: Optional[int], 真实标签索引（如果提供，会计算额外指标）
#     epsilon: float, 数值稳定性的小常数

#     返回:
#     Dict[str, float]: 包含各种不确定性指标的字典
#     """
#     # 输入验证
#     assert predictions.dim() == 2, "输入应该是2维张量 (n_estimates, n_classes)"
#     assert torch.all(predictions >= 0), "概率值不能为负数"
#     assert torch.allclose(predictions.sum(dim=-1), torch.tensor(1.0), atol=1e-6), (
#         "每行应该是归一化的概率分布"
#     )

#     n_estimates, n_classes = predictions.shape
#     if n_estimates < 2:
#         warnings.warn(f"采样数量过少({n_estimates})，不确定性估计可能不准确")

#     results = {}

#     # ========== 基础统计量 ==========
#     # 平均预测概率分布
#     mean_probs = predictions.mean(dim=0)

#     # 预测类别（概率最高的类别）
#     pred_class = mean_probs.argmax()
#     pred_class_probs = predictions[:, pred_class]

#     # 排序后的平均概率
#     sorted_probs, _ = torch.sort(mean_probs, descending=True)

#     # ========== 1. 信息论度量 ==========
#     # 1.1 预测熵 (总不确定性)
#     predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + epsilon))
#     results["predictive_entropy"] = predictive_entropy.item()

#     # 1.2 期望熵（偶然不确定性）
#     per_estimate_entropies = -torch.sum(
#         predictions * torch.log(predictions + epsilon), dim=1
#     )
#     expected_entropy = per_estimate_entropies.mean()
#     results["expected_entropy"] = expected_entropy.item()
#     results["aleatoric_uncertainty"] = expected_entropy.item()

#     # 1.3 互信息（认知不确定性）
#     mutual_information = predictive_entropy - expected_entropy
#     results["mutual_information"] = mutual_information.item()
#     results["epistemic_uncertainty"] = mutual_information.item()
#     results["bald_score"] = mutual_information.item()  # BALD = MI

#     # 1.4 条件熵
#     results["conditional_entropy"] = expected_entropy.item()

#     # ========== 2. 基于概率的度量 ==========
#     # 2.1 最大概率（置信度）
#     max_prob = mean_probs.max()
#     results["max_probability"] = max_prob.item()
#     results["confidence"] = max_prob.item()

#     # 2.2 置信度边际
#     if n_classes > 1:
#         margin = sorted_probs[0] - sorted_probs[1]
#         results["confidence_margin"] = margin.item()
#         results["margin_uncertainty"] = (1 - margin).item()

#         # 2.3 概率比率
#         ratio = sorted_probs[1] / (sorted_probs[0] + epsilon)
#         results["probability_ratio"] = ratio.item()

#     # 2.4 基尼不纯度
#     gini = 1 - torch.sum(mean_probs**2)
#     results["gini_impurity"] = gini.item()

#     # 2.5 归一化熵（0到1之间）
#     if n_classes > 1:
#         normalized_entropy = predictive_entropy / torch.log(
#             torch.tensor(n_classes, dtype=torch.float)
#         )
#         results["normalized_entropy"] = normalized_entropy.item()

#     # ========== 3. 基于方差的度量 ==========
#     # 3.1 预测类别的概率方差
#     pred_class_variance = pred_class_probs.var(unbiased=True)
#     results["pred_class_variance"] = pred_class_variance.item()

#     # 3.2 预测类别的标准差
#     pred_class_std = pred_class_probs.std(unbiased=True)
#     results["pred_class_std"] = pred_class_std.item()

#     # 3.3 预测类别的变异系数
#     pred_class_mean = pred_class_probs.mean()
#     coef_of_variation = pred_class_std / (pred_class_mean + epsilon)
#     results["pred_class_coef_of_variation"] = coef_of_variation.item()

#     # 3.4 所有类别的平均方差
#     all_class_variances = predictions.var(dim=0, unbiased=True)
#     results["mean_class_variance"] = all_class_variances.mean().item()
#     results["max_class_variance"] = all_class_variances.max().item()

#     # 3.5 最大概率的标准差
#     max_probs = predictions.max(dim=1)[0]
#     results["max_prob_std"] = max_probs.std(unbiased=True).item()

#     # ========== 4. 基于分布距离的度量 ==========
#     # 4.1 平均KL散度
#     kl_divs = []
#     for pred in predictions:
#         kl = torch.sum(pred * torch.log((pred + epsilon) / (mean_probs + epsilon)))
#         kl_divs.append(kl)
#     mean_kl = torch.mean(torch.stack(kl_divs))
#     results["mean_kl_divergence"] = mean_kl.item()

#     # 4.2 JS散度
#     js_div = 0
#     for pred in predictions:
#         m = (pred + mean_probs) / 2
#         kl1 = torch.sum(pred * torch.log((pred + epsilon) / (m + epsilon)))
#         kl2 = torch.sum(mean_probs * torch.log((mean_probs + epsilon) / (m + epsilon)))
#         js_div += (kl1 + kl2) / 2
#     results["js_divergence"] = (js_div / n_estimates).item()

#     # ========== 5. 基于采样一致性的度量 ==========
#     # 5.1 采样预测的一致性
#     pred_classes = predictions.argmax(dim=1)
#     mode_class, mode_count = torch.mode(pred_classes)
#     agreement_ratio = mode_count.float() / n_estimates
#     results["sample_agreement"] = agreement_ratio.item()
#     results["sample_disagreement"] = (1 - agreement_ratio).item()

#     # 5.2 预测多样性（不同预测类别的数量）
#     unique_predictions = torch.unique(pred_classes).numel()
#     results["prediction_diversity"] = unique_predictions / n_classes

#     # 5.3 熵的方差（衡量不同采样的熵的变化）
#     entropy_variance = per_estimate_entropies.var(unbiased=True)
#     results["entropy_variance"] = entropy_variance.item()

#     # ========== 6. 组合指标 ==========
#     # 6.1 不确定性综合分数（可根据需求调整权重）
#     uncertainty_score = (
#         0.3 * normalized_entropy.item()
#         if n_classes > 1
#         else 0.3 * predictive_entropy.item()
#         + 0.3 * (1 - max_prob.item())
#         + 0.2 * pred_class_variance.item()
#         + 0.2 * (1 - agreement_ratio.item())
#     )
#     results["uncertainty_score"] = uncertainty_score

#     # 6.2 可靠性指数
#     reliability_index = (
#         max_prob.item() * agreement_ratio.item() * (1 - pred_class_variance.item())
#     )
#     results["reliability_index"] = reliability_index

#     # ========== 7. 如果提供了真实标签的额外指标 ==========
#     # if true_label is not None:
#     #     assert 0 <= true_label < n_classes, f"真实标签 {true_label} 超出范围 [0, {n_classes})"

#     #     # 7.1 真实类别的概率统计
#     #     true_class_probs = predictions[:, true_label]
#     #     results["true_class_mean_prob"] = true_class_probs.mean().item()
#     #     results["true_class_std"] = true_class_probs.std(unbiased=True).item()
#     #     results["true_class_variance"] = true_class_probs.var(unbiased=True).item()

#     #     # 7.2 负对数似然
#     #     nll_samples = -torch.log(true_class_probs + epsilon)
#     #     results["mean_nll"] = nll_samples.mean().item()
#     #     results["nll_variance"] = nll_samples.var(unbiased=True).item()

#     #     # 7.3 Brier分数的不确定性
#     #     one_hot_true = torch.zeros(n_classes)
#     #     one_hot_true[true_label] = 1
#     #     brier_scores = torch.sum((predictions - one_hot_true) ** 2, dim=1)
#     #     results["mean_brier_score"] = brier_scores.mean().item()
#     #     results["brier_score_variance"] = brier_scores.var(unbiased=True).item()

#     # 7.4 预测是否正确的不确定性
#     # correct_predictions = (pred_classes == true_label).float()
#     # results["accuracy"] = correct_predictions.mean().item()
#     # results["prediction_stability"] = 1 - correct_predictions.var(unbiased=True).item()

#     # ========== 8. 额外的统计信息 ==========
#     # results["_num_estimates"] = n_estimates
#     # results["_num_classes"] = n_classes
#     # results["_predicted_class"] = pred_class.item()
#     # results["_predicted_prob"] = pred_class_mean.item()

#     return results
