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
from einops import asnumpy


def train(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
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
    model.train()
    teacher_model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, samples_teacher) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        samples_teacher = samples_teacher.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            outputs_teacher = teacher_model(samples_teacher)
            loss = criterion(outputs, outputs_teacher)

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
def evaluate(
    args, data_loader, model, teacher_model, criterion, device, epoch, mode, num_class
):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    loss_totals = []

    num_total = 0
    # switch to evaluation mode
    model.eval()

    criterion = torch.nn.MSELoss()

    for batch in metric_logger.log_every(data_loader, 10, header):
        image = batch[0]
        image_teacher = batch[-1]
        image = image.to(device, non_blocking=True)
        image_teacher = image_teacher.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(image)
            output_teacher = teacher_model(image_teacher)

            loss = criterion(output, output_teacher)

            data_bach = image.shape[0]
            num_total += data_bach

            for idx in range(data_bach):
                loss_totals.append(loss.item())

        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item())
        # metric_logger.meters["acc"].update(acc.item(), n=batch_size)

    metric_logger.synchronize_between_processes()

    if not os.path.exists(os.path.join(args.save_metrics_path)):
        os.makedirs(os.path.join(args.save_metrics_path))

    output_loss_total = np.mean(loss_totals)

    with open(
        os.path.join(args.save_metrics_path, "metrics.csv"),
        "a+",
    ) as txt:
        if epoch == 0:
            txt.write(f"Mode,Epoch,Loss\n")

        txt.write(f"{mode},{epoch},{output_loss_total}\n")

    print(f"{mode} Epoch {epoch}: Loss: {output_loss_total}\n")
    torch.cuda.empty_cache()
    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        output_loss_total,
    )
