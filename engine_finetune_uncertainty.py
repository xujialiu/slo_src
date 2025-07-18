import math
import sys
import os
import torch
import torch.nn.functional as F
from timm.data import Mixup
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np
from sklearn import metrics

# from util.loss import un_loss
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def kl_divergence(alpha, num_classes, device=None):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)

    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)

    y = F.one_hot(y, num_classes=num_classes)  # add
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def un_loss(
    target, alpha, num_classes, epoch_num, annealing_step, device=None, reduce=True
):
    if reduce:
        loss = torch.mean(
            edl_loss(
                torch.digamma,
                target,
                alpha,
                epoch_num,
                num_classes,
                annealing_step,
                device,
            )
        )
    else:
        loss = edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        ).squeeze(dim=1)
    return loss


class UncertaintyAwareLoss(torch.nn.Module):
    def __init__(self, lambda_entropy=0.1):
        super().__init__()
        self.lambda_entropy = lambda_entropy

    def forward(self, logits, labels):
        # 标准交叉熵损失
        ce_loss = F.cross_entropy(logits, labels)

        # 对DR类别（0-3）的预测添加熵正则化
        probs = F.softmax(logits, dim=1)
        dr_probs = probs[:, :4]  # 前4个类别的概率

        # 计算DR类别的熵（不包括uncertain类）
        dr_entropy = -torch.sum(dr_probs * torch.log(dr_probs + 1e-8), dim=1)

        # 对于标签为0-3的样本，如果熵很高，鼓励选择uncertain
        is_dr = (labels < 4).float()
        entropy_penalty = is_dr * dr_entropy

        total_loss = ce_loss - self.lambda_entropy * entropy_penalty.mean()
        return total_loss


class UncertaintyAwareLoss(torch.nn.Module):
    def __init__(self, lambda_entropy=0.1, reduction="mean"):
        super().__init__()
        self.lambda_entropy = lambda_entropy
        self.reduction = reduction

    def forward(self, logits, labels):
        # 标准交叉熵损失 - 不进行reduction
        ce_loss = F.cross_entropy(logits, labels, reduction="none")  # [batch]

        # 对DR类别（0-3）的预测添加熵正则化
        probs = F.softmax(logits, dim=1)
        dr_probs = probs[:, :4]  # 前4个类别的概率

        # 计算DR类别的熵（不包括uncertain类）
        dr_entropy = -torch.sum(dr_probs * torch.log(dr_probs + 1e-8), dim=1)  # [batch]

        # 对于标签为0-3的样本，如果熵很高，鼓励选择uncertain
        is_dr = (labels < 4).float()  # [batch]
        entropy_penalty = is_dr * dr_entropy  # [batch]

        # 计算每个样本的总损失
        total_loss = ce_loss - self.lambda_entropy * entropy_penalty  # [batch]

        # 根据reduction参数决定返回格式
        if self.reduction == "none":
            return total_loss  # [batch]
        elif self.reduction == "mean":
            return total_loss.mean()  # 标量
        elif self.reduction == "sum":
            return total_loss.sum()  # 标量
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


def train(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(
            data_loader,
            print_freq,
            header,
        )
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)

            if args.loss_type == "cross_entropy":
                loss = criterion(outputs, targets)

            elif args.loss_type == "uncertainty":
                evidence = F.softplus(outputs)
                alpha = evidence + 1

                S = torch.sum(alpha, dim=1, keepdim=True)
                E = alpha - 1
                prob = E / (S.expand(E.shape))

                loss_ce = criterion(prob, targets)

                loss_un = un_loss(
                    targets, alpha, args.nb_classes, epoch, args.epochs, device
                )
                loss = loss_ce + loss_un

            elif args.loss_type == "uncertainty_new":
                evidence = F.softplus(outputs)
                alpha = evidence + 1

                S = torch.sum(alpha, dim=1, keepdim=True)
                E = alpha - 1
                prob = E / (S.expand(E.shape))

                # loss_ce = criterion(prob, targets)

                loss_un = un_loss(
                    targets,
                    alpha,
                    args.nb_classes,
                    epoch,
                    args.epochs,
                    device,
                )
                loss = loss_un

            elif args.loss_type == "my_uncertainty":
                criterion = UncertaintyAwareLoss()
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, data_loader, model, device, epoch, mode, num_class):
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    labels = []
    outputs = []

    predictions = []
    ground_truths = []
    uncertainties = []
    probabilities = []

    loss_ces = []
    loss_totals = []
    loss_uns = []

    num_total = 0
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        image = batch[0]
        target = batch[-1]
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(image)

            if args.loss_type == "cross_entropy":
                loss_un = torch.tensor([0] * args.batch_size)
                uncertainty = torch.tensor([0] * args.batch_size)

                loss_ce = criterion(output, target)
                loss = loss_ce
                pred = F.softmax(output, dim=-1)

            elif args.loss_type == "uncertainty":
                evidence = F.softplus(output)
                alpha = evidence + 1

                S = torch.sum(alpha, dim=1, keepdim=True)
                E = alpha - 1
                pred = E / (S.expand(E.shape))
                # shape: [batch_size, nb_classes], train中的b相当于pred

                # 计算uncertainty, added by xujia
                # 使用Dirichlet distribution的variance作为uncertainty度量
                # uncertainty = K/(S * (S + 1)), K是类别数
                K = torch.tensor(args.nb_classes).to(device, non_blocking=True)
                uncertainty = K / S  # shape: [batch_size, 1]

                loss_ce = criterion(pred, target)

                loss_un = un_loss(
                    target,
                    alpha,
                    args.nb_classes,
                    epoch,
                    args.epochs,
                    device,
                    reduce=False,
                )
                loss = loss_ce + loss_un

            elif args.loss_type == "uncertainty_new":
                evidence = F.softplus(output)
                alpha = evidence + 1

                S = torch.sum(alpha, dim=1, keepdim=True)
                E = alpha - 1
                pred = E / (S.expand(E.shape))
                # shape: [batch_size, nb_classes], train中的b相当于pred

                # 计算uncertainty, added by xujia
                # 使用Dirichlet distribution的variance作为uncertainty度量
                # uncertainty = K/(S * (S + 1)), K是类别数
                K = torch.tensor(args.nb_classes).to(device, non_blocking=True)
                uncertainty = (K / S).squeeze(1)  # shape: [batch_size, 1]

                loss_un = un_loss(
                    target,
                    alpha,
                    args.nb_classes,
                    epoch,
                    args.epochs,
                    device,
                    reduce=False,
                )
                loss = loss_un
                loss_ce = loss_un

            elif args.loss_type == "my_uncertainty":
                criterion = UncertaintyAwareLoss(reduction="none")
                loss = criterion(output, target)
                loss_un = loss_ce = loss
                uncertainty = F.softmax(output, dim=-1)[:, -1]  # uncertain类的概率
                pred = F.softmax(output, dim=-1)

            data_bach = image.shape[0]
            num_total += data_bach

            one_hot = (
                torch.zeros(data_bach, args.nb_classes)
                .to(device)
                .scatter_(1, target.unsqueeze(1), 1)
            )
            pred_decision = output.argmax(dim=-1)

            pred_softmax = F.softmax(output, dim=-1)  # 方便后续计算
            # breakpoint()
            # add loss_CE, loss_un, loss_total to excel, xujia
            for idx in range(data_bach):
                loss_ces.append(loss_ce.cpu().detach().numpy()[idx])
                loss_totals.append(loss.cpu().detach().numpy()[idx])
                loss_uns.append(loss_un.cpu().detach().numpy()[idx])

                outputs.append(pred_softmax.cpu().detach().numpy()[idx])
                labels.append(one_hot.cpu().detach().numpy()[idx])
                predictions.append(pred_decision.cpu().detach().numpy()[idx])
                ground_truths.append(target.cpu().detach().numpy()[idx])
                uncertainties.append(uncertainty.cpu().detach().numpy()[idx])
                probabilities.append(pred.cpu().detach().numpy()[idx])

        acc = metrics.accuracy_score(
            target.cpu().detach().numpy(), pred_decision.cpu().detach().numpy()
        )

        batch_size = image.shape[0]
        metric_logger.update(loss=loss.mean().item())
        metric_logger.meters["acc"].update(acc.item(), n=batch_size)

    metric_logger.synchronize_between_processes()

    if not os.path.exists(os.path.join(args.save_metrics_path)):
        os.makedirs(os.path.join(args.save_metrics_path))
    # save results, xujia
    pd.DataFrame(
        {
            "labels": labels,
            "outputs": outputs,
            "ground_truths": ground_truths,
            "predictions": predictions,
            "probabilities": probabilities,
            "uncertainties": uncertainties,
            "loss_totals": loss_totals,
            "loss_CEs": loss_ces,
            "loss_uns": loss_uns,
        }
    ).to_csv(os.path.join(args.save_metrics_path, f"results_{epoch}.csv"))

    if args.nb_classes > 2:
        try:
            auc_weighted = metrics.roc_auc_score(
                ground_truths, outputs, multi_class="ovr", average="weighted"
            )
        except ValueError:
            auc_weighted = 0.0

        try:
            auc_macro = metrics.roc_auc_score(
                ground_truths, outputs, multi_class="ovr", average="macro"
            )
        except ValueError:
            auc_macro = 0.0

        accuracy = metrics.accuracy_score(ground_truths, predictions)
        precision = metrics.precision_score(
            ground_truths, predictions, average="weighted"
        )
        sensitivity = metrics.recall_score(
            ground_truths, predictions, average="weighted"
        )
        specificity = weighted_specificity(ground_truths, predictions)
        f1 = metrics.f1_score(ground_truths, predictions, average="weighted")
        kappa = metrics.cohen_kappa_score(
            ground_truths, predictions, weights="quadratic"
        )
    else:
        outputs = [output[1] for output in outputs]
        try:
            auc_weighted = metrics.roc_auc_score(
                ground_truths,
                outputs,
            )
        except ValueError:
            auc_weighted = 0.0

        try:
            auc_macro = metrics.roc_auc_score(
                ground_truths,
                outputs,
            )
        except ValueError:
            auc_macro = 0.0

        accuracy = metrics.accuracy_score(ground_truths, predictions)
        precision = metrics.precision_score(ground_truths, predictions)
        sensitivity = metrics.recall_score(ground_truths, predictions)
        specificity = weighted_specificity(ground_truths, predictions)
        f1 = metrics.f1_score(ground_truths, predictions)
        kappa = metrics.cohen_kappa_score(ground_truths, predictions)

    output_loss_total = np.mean(loss_totals)
    output_loss_un = np.mean(loss_uns)
    output_loss_CE = np.mean(loss_ces)

    cm_path = os.path.join(args.save_metrics_path, f"cm_{epoch}.png")
    plot_cm(ground_truths, predictions, cm_path)

    with open(
        os.path.join(args.save_metrics_path, "metrics.csv"),
        "a+",
    ) as txt:
        if epoch == 0:
            txt.write(
                f"Mode,Epoch,AUC_macro,AUC_weighted,F1,Kappa,Accuracy,Precision,Sensitivity,Specificity,Loss\n"
            )

        txt.write(
            f"{mode},{epoch},{auc_macro},{auc_weighted},{f1},{kappa},{accuracy},{precision},{sensitivity},{specificity},{output_loss_total}\n"
        )

    print(
        f"{mode} Epoch {epoch}: AUC macro: {auc_macro}, AUC weighted: {auc_weighted}, F1: {f1}, Kappa: {kappa}, Accuracy: {accuracy}, Precision: {precision}, Sensitivity: {sensitivity}, Specificity: {specificity}, Loss: {output_loss_total}\n"
    )
    torch.cuda.empty_cache()
    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        auc_weighted,
        accuracy,
        (output_loss_total, output_loss_un, output_loss_CE),
    )


def plot_cm(ground_truths, predictions, save_path):
    cm = confusion_matrix(ground_truths, predictions)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Percentage)")

    fig.savefig(save_path)


def weighted_specificity(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred)

    n_classes = cm.shape[0]

    specificities = []
    class_weights = []

    for i in range(n_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm[i, :], i))

        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        specificities.append(specificity)

        class_weights.append(np.sum(y_true == i))

    class_weights = np.array(class_weights) / len(y_true)
    weighted_avg = np.sum(np.array(specificities) * class_weights)

    return weighted_avg
