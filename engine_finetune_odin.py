import math
from pathlib import Path
import sys
import os
from einops import asnumpy
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
from timm.data.constants import IMAGENET_DEFAULT_STD


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
                    targets, alpha, args.nb_classes, epoch, args.epochs, device
                )
                loss = loss_un

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
                loss = loss_un
                loss_ce = loss_un

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


def test(args, data_loader, model, device, epoch, mode, num_class):

    TEMP = 1000
    NOISE_MAGNITUDE = 0.0014
    std1, std2, std3 = IMAGENET_DEFAULT_STD

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    list_labels = []
    list_predictions = []
    list_probabilities = []
    list_confidences_ori = []
    list_confidences_pertubed = []

    num_total = 0
    # switch to evaluation mode
    model.eval()

    for idx, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # if idx > 100:
        #     break

        image = batch[0]
        target = batch[-1]

        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        image.requires_grad_(True)

        # compute output
        with torch.cuda.amp.autocast():
            output_ori = model(image)

            if image.grad is not None:
                image.grad.zero_()

            output_ori_np = asnumpy(output_ori)[0]
            output_ori_np = output_ori_np - np.max(output_ori_np)
            output_ori_np_softmax = np.exp(output_ori_np) / np.sum(
                np.exp(output_ori_np)
            )
            confidence_ori = np.max(output_ori_np_softmax)

            # 计算伪标签
            output_ori = output_ori / TEMP
            psedo_label = np.argmax(output_ori_np)
            psedo_label = (
                torch.tensor(psedo_label).unsqueeze(0).to(device, non_blocking=True)
            )

            loss_pseudo = criterion(output_ori, psedo_label)
            loss_pseudo.backward()

            gradient = image.grad.data.clone()
            gradient = image.grad.data > 0
            gradient = (gradient.float() - 0.5) * 2

            gradient[0][0] = (gradient[0][0]) / std1  # 除以标准差
            gradient[0][1] = (gradient[0][1]) / std2
            gradient[0][2] = (gradient[0][2]) / std3

            # perturbed input
            image_perturbed = image - NOISE_MAGNITUDE * gradient
            output_perturbed = model(image_perturbed)
            output_perturbed_np = asnumpy(output_perturbed)[0]
            output_perturbed_np = output_perturbed_np - np.max(output_perturbed_np)
            output_perturbed_np = output_perturbed_np / TEMP
            output_perturbed_np_softmax = np.exp(output_perturbed_np) / np.sum(
                np.exp(output_perturbed_np)
            )
            confidence_perturbed = np.max(output_perturbed_np_softmax)

            prob = asnumpy(F.softmax(output_ori[0], dim=-1))  # 方便后续计算
            pred = np.argmax(prob)

            list_labels.append(asnumpy(target)[0])
            list_predictions.append(pred)
            list_probabilities.append(prob)
            list_confidences_ori.append(confidence_ori)
            list_confidences_pertubed.append(confidence_perturbed)

    # save results, xujia
    df = pd.DataFrame(
        {
            "labels": list_labels,
            "predictions": list_predictions,
            "probabilities": list_probabilities,
            "confidences_ori": list_confidences_ori,
            "confidences_pertubed": list_confidences_pertubed,
        }
    )

    df["delta_confidence"] = df["confidences_ori"] - df["confidences_pertubed"]

    if not Path(args.save_metrics_path).exists():
        Path(args.save_metrics_path).mkdir(parents=True, exist_ok=True)

    df.to_csv(os.path.join(args.save_metrics_path, f"results_{epoch}.csv"))


def test(args, data_loader, model, device, epoch, mode, num_class):
    TEMP = 100.0
    NOISE_MAGNITUDE = 0.01
    std_tensor = (
        torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1).to(device)
    )  # 转为张量

    model.eval()
    results = []

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    for idx, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # if idx > 100:
        #     break

        images, targets = batch[0].to(device), batch[-1].to(device)
        batch_size = images.shape[0]

        # 启用梯度
        images.requires_grad_(True)

        # 原始输出
        with torch.cuda.amp.autocast():
            outputs = model(images)

        # 温度缩放 + softmax
        scaled_outputs = outputs / TEMP
        probs = torch.softmax(scaled_outputs, dim=1)
        conf_ori = probs.max(1).values

        # 计算伪标签梯度
        pseudo_labels = scaled_outputs.argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(scaled_outputs, pseudo_labels)
        loss.backward()

        # 获取梯度符号 + 归一化
        gradient_sign = torch.sign(images.grad.data)
        normalized_grad = gradient_sign / std_tensor

        # 应用扰动 (ODIN: x' = x - ε·sign(∇x))
        perturbed_images = images - NOISE_MAGNITUDE * normalized_grad
        perturbed_images.detach_()

        # 扰动后预测
        with torch.cuda.amp.autocast(), torch.no_grad():
            perturbed_outputs = model(perturbed_images)

        # 温度缩放 + softmax
        scaled_perturbed = perturbed_outputs / TEMP
        perturbed_probs = torch.softmax(scaled_perturbed, dim=1)
        conf_perturbed = perturbed_probs.max(1).values

        # 保存结果
        for i in range(batch_size):
            results.append(
                {
                    "label": targets[i].item(),
                    "prediction": pseudo_labels[i].item(),
                    "confidence_ori": conf_ori[i].item(),
                    "confidence_perturbed": conf_perturbed[i].item(),
                    "delta_confidence": (conf_ori[i] - conf_perturbed[i]).item(),
                }
            )

    # 保存结果
    df = pd.DataFrame(results)
    os.makedirs(args.save_metrics_path, exist_ok=True)
    df.to_csv(os.path.join(args.save_metrics_path, f"results_{epoch}.csv"))


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
