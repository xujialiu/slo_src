import math
from pathlib import Path
import sys
import os
from einops import asnumpy
import torch
import torch.nn.functional as F
from timm.data import Mixup
from typing import Iterable, Optional
import util.misc_energy as misc
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
    data_loader_in: Iterable,
    data_loader_out: Iterable,
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

    len_dataset = len(data_loader_in)
    for data_iter_step, ((in_samples, targets), out_samples) in enumerate(
        metric_logger.log_every(
            zip(data_loader_in, data_loader_out),
            len_dataset,
            print_freq,
            header,
        )
    ):
        # if data_iter_step > 10:
        #     break

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader_in) + epoch, args
            )

        samples = torch.cat((in_samples, out_samples), 0)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        num_in_samples = in_samples.shape[0]

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)

            # energy ood
            loss = F.cross_entropy(outputs[:num_in_samples], targets)

            Ec_out = -torch.logsumexp(outputs[num_in_samples:], dim=1)
            Ec_in = -torch.logsumexp(outputs[:num_in_samples], dim=1)
            loss += 0.1 * (
                torch.pow(F.relu(Ec_in - args.m_in), 2).mean()
                + torch.pow(F.relu(args.m_out - Ec_out), 2).mean()
            )

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
            epoch_1000x = int((data_iter_step / num_in_samples + epoch) * 1000)
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
    scores = []
    probabilities = []

    loss_ces = []
    loss_totals = []
    loss_uns = []

    num_total = 0
    # switch to evaluation mode
    model.eval()

    len_dataset = len(data_loader)
    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, len_dataset, 10, header)
    ):
        # if data_iter_step > 10:
        #     break

        image = batch[0]
        target = batch[-1]
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(image)

            score = -asnumpy(args.T * torch.logsumexp(output / args.T, dim=1))
            loss_ce = criterion(output, target)
            pred = F.softmax(output, dim=-1)
            score = score
            loss = loss_ce

            data_bach = image.shape[0]
            num_total += data_bach

            one_hot = (
                torch.zeros(data_bach, args.nb_classes)
                .to(device)
                .scatter_(1, target.unsqueeze(1), 1)
            )
            pred_decision = output.argmax(dim=-1)

            pred_softmax = F.softmax(output, dim=-1)  # 方便后续计算
            # add loss_CE, loss_un, loss_total to excel, xujia
            for idx in range(data_bach):
                loss_ces.append(loss_ce.cpu().detach().numpy()[idx])
                loss_totals.append(loss.cpu().detach().numpy()[idx])

                outputs.append(pred_softmax.cpu().detach().numpy()[idx])
                labels.append(one_hot.cpu().detach().numpy()[idx])
                predictions.append(pred_decision.cpu().detach().numpy()[idx])
                ground_truths.append(target.cpu().detach().numpy()[idx])
                scores.append(score[idx])
                probabilities.append(pred.cpu().detach().numpy()[idx])

        acc = metrics.accuracy_score(
            target.cpu().detach().numpy(), pred_decision.cpu().detach().numpy()
        )

        batch_size = image.shape[0]
        metric_logger.update(loss=loss.mean().item())
        metric_logger.meters["acc"].update(acc, n=batch_size)

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
            "scores": scores,
            "loss_CEs": loss_ces,
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
            f"{mode},{epoch},{auc_macro},{auc_weighted},{f1},{kappa},{accuracy},{precision},{sensitivity},{specificity},{output_loss_CE}\n"
        )

    print(
        f"{mode} Epoch {epoch}: AUC macro: {auc_macro}, AUC weighted: {auc_weighted}, F1: {f1}, Kappa: {kappa}, Accuracy: {accuracy}, Precision: {precision}, Sensitivity: {sensitivity}, Specificity: {specificity}, Loss: {output_loss_CE}\n"
    )
    torch.cuda.empty_cache()
    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        auc_weighted,
        accuracy,
        (output_loss_CE, output_loss_CE, output_loss_CE),
    )


@torch.no_grad()
def evaluate_out(args, data_loader, model, device, epoch):
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    labels = []
    outputs = []

    predictions = []
    ground_truths = []
    scores = []
    probabilities = []

    loss_ces = []
    loss_totals = []
    loss_uns = []

    num_total = 0
    # switch to evaluation mode
    model.eval()

    len_dataset = len(data_loader)
    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, len_dataset, 10, header)
    ):
        # if data_iter_step > 10:
        #     break

        image = batch
        image = image.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(image)

            score = -asnumpy(args.T * torch.logsumexp(output / args.T, dim=1))
            pred = F.softmax(output, dim=-1)
            score = score

            data_bach = image.shape[0]
            num_total += data_bach

            # add loss_CE, loss_un, loss_total to excel, xujia
            for idx in range(data_bach):
                scores.append(score[idx])

        batch_size = image.shape[0]

    metric_logger.synchronize_between_processes()

    if not os.path.exists(os.path.join(args.save_metrics_path)):
        os.makedirs(os.path.join(args.save_metrics_path))
    # save results, xujia
    pd.DataFrame(
        {
            "scores": scores,
        }
    ).to_csv(os.path.join(args.save_metrics_path, f"results_out_{epoch}.csv"))

    torch.cuda.empty_cache()


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
