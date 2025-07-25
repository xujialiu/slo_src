import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup

# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from peft import LoraConfig, get_peft_model

import vit_model_new

from engine_finetune import train, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser(
        "MAE fine-tuning for image classification", add_help=True
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input_size", default=448, type=int, help="images input size")

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=2e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=10, metavar="N", help="epochs to warmup LR"
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # * Finetuning params
    parser.add_argument(
        "--finetune",
        default="./foundation_model_weights/pretrain_distill_ffm_checkpoint-best.pth",
        type=str,
        help="finetune from checkpoint",
    )
    parser.add_argument(
        "--task", default="FMUE", type=str, help="finetune from checkpoint"
    )
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    # Dataset parameters
    parser.add_argument(
        "--nb_classes", default=5, type=int, help="number of the classification types"
    )
    parser.add_argument(
        "--output_dir",
        default="checkpoints",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="log", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    # parser.add_argument(
    #     "--pin_mem",
    #     action="store_true",
    #     help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    # )
    # parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    # parser.set_defaults(pin_mem=True)
    parser.add_argument(
        "--no_pin_mem",
        action="store_false",
        default=True,
        dest="pin_mem",
        help="Disable pinning CPU memory in DataLoader for GPU transfer.",
    )

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument(
        "--fulltune", default=False, action="store_true", help="full finetune model"
    )
    parser.add_argument(
        "--metrics_folder",
        default="metrics",
        help="path where to save metrics, empty for no saving",
    )
    parser.add_argument(
        "--net_work", default="FMUE", help="path where to save, empty for no saving"
    )

    # dataset parameters
    parser.add_argument("--csv_path", help="path where csv file is located")
    parser.add_argument("--data_path", help="path where images are located")

    # results parameters
    parser.add_argument(
        "--result_root_path",
        default="./results",
        help="path where results will be saved",
    )
    parser.add_argument(
        "--result_name",
        default=f"SLO",
        help="path where results will be saved",
    )

    # weights
    parser.add_argument(
        "--class_percentages",
        type=json.loads,
        default="[0.354835, 0.164789,0.345921,0.061161, 0.073295]",
        help='A list of numbers, e.g., "[0.354835, 0.164789,0.345921,0.061161, 0.073295]"',
    )

    return parser


def main(args):

    print("{}".format(args).replace(", ", ",\n"))

    args.result_name = f"{args.result_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    result_path = Path(args.result_root_path) / args.result_name

    args.output_dir = str(result_path / args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.save_metrics_path = str(result_path / args.metrics_folder)

    misc.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train="train", args=args)
    dataset_val = build_dataset(is_train="test", args=args)
    dataset_test = build_dataset(is_train="test", args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        log_path = result_path / args.log_dir
        log_writer = SummaryWriter(log_dir=log_path)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    model = vit_model_new.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location="cpu")

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    # load peft model
    config_qkv = LoraConfig(
        r=4,  # LoRA的秩
        lora_alpha=8,  # LoRA的alpha参数, scaling=alpha/r
        target_modules=["qkv"],  # 需要应用LoRA的模块名称
        lora_dropout=0,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    get_peft_model(model, config_qkv)

    # freeze the pretrained weights
    for name, param in model.named_parameters():
        if "head" in name or "lora" in name or "fc_norm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # 当没有设置为fulltune时
    if args.fulltune:
        print("Full finetune model")
        for name, param in model.named_parameters():
            param.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    # classes weights
    class_weights = 1 / torch.FloatTensor(args.class_percentages).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print("criterion = %s" % str(criterion))

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    if args.eval:
        (
            test_stats,
            test_auc_roc,
            test_acc,
            (test_output_loss_total, test_output_loss_un, test_output_loss_CE),
        ) = evaluate(
            data_loader_test,
            model,
            device,
            epoch=0,
            mode="test",
            num_class=args.nb_classes,
        )
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_auc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            mixup_fn,
            log_writer=log_writer,
            args=args,
        )

        (
            val_stats,
            val_auc_roc,
            val_acc,
            (val_output_loss_total, val_output_loss_un, val_output_loss_CE),
        ) = evaluate(
            args,
            data_loader_val,
            model,
            device,
            epoch,
            mode="val",
            num_class=args.nb_classes,
        )
        if args.output_dir:
            misc.save_model_epoch(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
        if max_auc < val_auc_roc:
            max_auc = val_auc_roc

            if args.output_dir:
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )

        if log_writer is not None:
            log_writer.add_scalar("perf/val_acc1", val_stats["acc1"], epoch)
            log_writer.add_scalar("perf/val_auc", val_auc_roc, epoch)
            log_writer.add_scalar("perf/val_loss", val_stats["loss"], epoch)

            # add by xujia
            log_writer.add_scalar("perf/val_total_loss", val_output_loss_total, epoch)
            log_writer.add_scalar("perf/val_loss_un", val_output_loss_un, epoch)
            log_writer.add_scalar("perf/val_loss_CE", val_output_loss_CE, epoch)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
