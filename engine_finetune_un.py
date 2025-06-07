import math
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
from pycm import *
import numpy as np
from sklearn import metrics
from util.loss import un_loss
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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
        metric_logger.log_every(data_loader, print_freq, header)
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
            evidence = F.softplus(outputs)
            alpha = evidence + 1

            S = torch.sum(alpha, dim=1, keepdim=True)
            E = alpha - 1
            prob = E / (S.expand(E.shape))

            loss_CE = criterion(prob, targets)

            loss_un = un_loss(
                targets, alpha, args.nb_classes, epoch, args.epochs, device
            )
            loss = loss_CE + loss_un

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
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    labels = []
    outputs = []

    predictions = []
    ground_truths = []
    uncertainties = []
    probabilities = []

    loss_CEs = []
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

            loss_CE = criterion(pred, target)

            loss_un = un_loss(
                target, alpha, args.nb_classes, epoch, args.epochs, device
            )
            loss = loss_CE + loss_un

            data_bach = pred.shape[0]
            num_total += data_bach

            one_hot = (
                torch.zeros(data_bach, args.nb_classes)
                .to(device)
                .scatter_(1, target.unsqueeze(1), 1)
            )
            pred_decision = pred.argmax(dim=-1)

            pred_softmax = F.softmax(pred, dim=-1)  # 方便后续计算

            # add loss_CE, loss_ACE, loss_un, loss_total to excel, xujia
            loss_CEs.append(loss_CE.item())
            loss_totals.append(loss.item())
            loss_uns.append(loss_un.item())

            for idx in range(data_bach):
                outputs.append(pred_softmax.cpu().detach().float().numpy()[idx])
                labels.append(one_hot.cpu().detach().float().numpy()[idx])
                predictions.append(pred_decision.cpu().detach().float().numpy()[idx])
                ground_truths.append(target.cpu().detach().float().numpy()[idx])
                uncertainties.append(uncertainty.cpu().detach().float().numpy()[idx][0])
                probabilities.append(pred_softmax.cpu().detach().float().numpy()[idx])

        acc1, _ = accuracy(output, target, topk=(1, 2))

        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

    metric_logger.synchronize_between_processes()

    epoch_auc = metrics.roc_auc_score(
        ground_truths, outputs, multi_class="ovr", average="weighted"
    )
    Acc = metrics.accuracy_score(ground_truths, predictions)

    if not os.path.exists(os.path.join(args.save_metrics_path)):
        os.makedirs(os.path.join(args.save_metrics_path))

    output_loss_total = np.mean(loss_totals)
    output_loss_un = np.mean(loss_uns)
    output_loss_CE = np.mean(loss_CEs)

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
            "loss_CEs": loss_CEs,
            "loss_uns": loss_uns,
        }
    ).to_excel(os.path.join(args.save_metrics_path, f"results_{epoch}.xlsx"))

    cm_path = os.path.join(args.save_metrics_path, f"cm_{epoch}.png")
    plot_cm(ground_truths, predictions, cm_path)

    with open(
        os.path.join(args.save_metrics_path, "metrics.txt"),
        "a+",
    ) as txt:
        txt.write(f"{mode} Epoch {epoch}: Acc: {Acc}, AUC: {epoch_auc}\n")

    print(f"{mode} Epoch {epoch}: Acc: {Acc}, AUC: {epoch_auc}\n")
    torch.cuda.empty_cache()
    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        epoch_auc,
        Acc,
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
