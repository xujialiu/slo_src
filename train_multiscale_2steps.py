import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
import ast
from easydict import EasyDict


import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from peft import LoraConfig, get_peft_model

# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay_multiscale as lrd
import util.misc as misc
from util.datasets_multiscale import build_dataset, get_weighted_sampler
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_finetune import train, evaluate

# import vit_model_patches
from model import MultiScale

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`torch.cuda.amp.autocast.*` is deprecated",
)


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
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_large_patch16_patches",
        type=str,
        metavar="MODEL",
        help="""Name of model to train 
                (vit_large_patch16_multiscales, vit_base_patch16_multiscales, 
                 vit_large_patch16_patches, vit_base_patch16_patches, 
                 vit_large_patch16, vit_base_patch16, 
                 default: vit_large_patch16_patches)""",
    )
    parser.add_argument(
        "--input_size",
        default=[896, 896],
        type=ast.literal_eval,
        help="model input size ([width,height], default: [896,896])",
    )

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.2,
        metavar="PCT",
        help="Drop path rate (default: 0.2)",
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
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256 (default: 1e-3)",
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
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
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
        default=None,
        type=str,
        help="finetune from checkpoint",
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
        "--device", default="cuda:0", help="device to use for training / testing"
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
        help="Enabling distributed evaluation (recommended during training for faster monitor)",
    )
    parser.add_argument("--num_workers", default=8, type=int)

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
        "--fulltune",
        default=False,
        action="store_true",
        help="full finetune model (default: False)",
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
    parser.add_argument("--dataset_type", default="slo", help="dataset type")

    # results parameters
    parser.add_argument(
        "--result_root_path",
        default="./results",
        help="path where results will be saved",
    )
    parser.add_argument(
        "--result_name",
        default="SLO",
        help="path where results will be saved",
    )

    # lora
    parser.add_argument(
        "--lora_position",
        default="backbone",
        type=str,
        help="position of lora layer (backbone or all, default: backbone)",
    )
    parser.add_argument(
        "--lora_bias",
        default="lora_only",
        type=str,
        help="bias of lora layer (none, all, lora_only, default: lora_only)",
    )
    parser.add_argument(
        "--lora_dropout",
        default=0.1,
        type=float,
        help="dropout rate of lora layer (default: 0.1)",
    )
    parser.add_argument(
        "--lora_rank",
        default=4,
        type=int,
        help="lora rank (default: 4)",
    )
    parser.add_argument(
        "--lora_alpha",
        default=8,
        type=int,
        help="lora alpha (default: 8)",
    )

    # loss function
    parser.add_argument(
        "--loss_type",
        default="cross_entropy",
        type=str,
        help="loss function type (cross_entropy or uncertainty, default: cross_entropy)",
    )

    # fusion block
    parser.add_argument(
        "--fusion_layer_num",
        default=2,
        type=int,
        help="number of fusion layers (default: 2)",
    )
    parser.add_argument(
        "--fusion_dropout",
        default=0.1,
        type=float,
        help="dropout rate of fusion layers (default: 0.1)",
    )
    parser.add_argument(
        "--use_learnable_pos_embed",
        default=False,
        action="store_true",
        help="use learnable position embedding (default: False)",
    )

    # dataset
    parser.add_argument(
        "--random_crop_perc",
        default=0.9,
        type=float,
        help="random crop percentage (default: 0.9)",
    )
    parser.add_argument(
        "--resize_to",
        default="[3072,3900]",
        type=ast.literal_eval,
        help="resize image size for uniformize image size ([width,height], default: [3072,3900])",
    )
    parser.add_argument(
        "--use_weighted_sampler",
        default=True,
        type=lambda x: x.lower() == "true",
        help="use weighted sampler or not (default: True)",
    )

    # model
    parser.add_argument(
        "--fm_input_size",
        default="448",
        type=int,
        help="input size of foundation model, works only for vit_large_patch16_patches ([width,height], default: [448,448])",
    )

    # cross entropy weight
    parser.add_argument(
        "--use_cross_entropy_weight",
        default=False,
        type=lambda x: x.lower() == "true",
        help="use cross entropy weight or not (default: False)",
    )

    # llrd
    parser.add_argument(
        "--use_llrd",
        default=False,
        action="store_true",
        help="Layer-wise learning rate decay in LoRA (default: False)",
    )

    # multiscales
    parser.add_argument(
        "--num_patches",
        default=None,
        type=ast.literal_eval,
        help="""number of patches for multiscale training, 
                  work with vit_large_patch16_multiscales and vit_base_patch16_multiscales 
                  (default: None)""",
    )

    # train_all
    parser.add_argument(
        "--train_all",
        default=False,
        action="store_true",
        help="using all samples for training (default: False)",
    )

    # early stopping
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Number of epochs to wait before early stopping (0 means disabled)",
    )
    parser.add_argument(
        "--early_stop_monitor",
        type=str,
        default="val_auc",
        choices=["val_auc", "val_loss"],
        help="Metric to monitor for early stopping",
    )

    # training step
    parser.add_argument(
        "--train_step",
        type=int,
        default=1,
        choices=[1, 2],
        help="Metric to monitor for early stopping",
    )

    return parser


def main(args):
    print(f"{args}".replace(", ", ",\n"))

    args.result_name = f"{args.result_name}_{datetime.now().strftime(r'%Y%m%d_%H%M')}"
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

    if args.train_all:
        dataset_train = build_dataset(is_train="all", args=args)
        dataset_val = build_dataset(is_train="all", args=args)
    else:
        dataset_train = build_dataset(is_train="train", args=args)
        dataset_val = build_dataset(is_train="val", args=args)

    if args.eval:
        dataset_test = build_dataset(is_train="test", args=args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # create train sampler
    if args.use_weighted_sampler:
        try:
            sampler_train = get_weighted_sampler(dataset_train)
            print(f"use weighted sampler")
        except:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True,
            )
        print(f"use distributed sampler")
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print(f"use distributed sampler")
    print(f"Sampler_train = {sampler_train}")

    # create val sampler
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print(
                "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # create test sampler
    if args.dist_eval and args.eval:
        if len(dataset_test) % num_tasks != 0:
            print(
                "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    elif args.eval:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        log_path = result_path / args.log_dir
        log_writer = SummaryWriter(log_dir=log_path)
    else:
        log_writer = None

    # create train and val data loaders
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    if args.eval:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=args.batch_size,
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

    if "large" in args.model:
        kwargs = EasyDict(
            embed_dim=1024,
            depth=24,
            num_heads=16,
        )
    elif "base" in args.model:
        kwargs = EasyDict(
            embed_dim=768,
            depth=12,
            num_heads=12,
        )

    model = MultiScale(
        input_size=args.input_size,
        num_patches=args.num_patches,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        fusion_layer_num=args.fusion_layer_num,
        fusion_dropout=args.fusion_dropout,
        fm_input_size=args.fm_input_size,
        use_learnable_pos_embed=args.use_learnable_pos_embed,
        **kwargs,
    )

    if args.lora_position == "all":
        target_modules = ["qkv"]
    elif args.lora_position == "backbone":
        target_modules = "image_encoder\.blocks\.\d+\.attn\.qkv$"

    # load pre-trained model
    # 当finetune和resume都没有设置时
    if (not args.finetune) and (not args.resume):
        raise ValueError("Please specify a pre-trained model for finetuning")
    # 使用finetune的情况
    elif (not args.eval) and (not args.resume):
        checkpoint = torch.load(args.finetune, map_location="cpu")

        print(f"Load pre-trained checkpoint from: {args.finetune}")
        checkpoint_model = checkpoint["model"]
        if args.train_step == 1:
            checkpoint_model = {
                "image_encoder." + key: value for key, value in checkpoint_model.items()
            }
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f"{msg=}")

        if args.train_step == 1:
            # manually initialize fc layer
            trunc_normal_(model.head.weight, std=2e-5)

        if (not args.fulltune) and (args.train_step == 2):
            # load peft model
            config_qkv = LoraConfig(
                r=args.lora_rank,  # LoRA的秩
                lora_alpha=args.lora_alpha,  # LoRA的alpha参数, scaling=alpha/r
                target_modules=target_modules,  # 需要应用LoRA的模块名称
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                task_type="FEATURE_EXTRACTION",
            )
            get_peft_model(model, config_qkv)
    # 使用resume的情况
    elif args.resume:
        if not args.fulltune:
            # load peft model
            config_qkv = LoraConfig(
                r=args.lora_rank,  # LoRA的秩
                lora_alpha=args.lora_alpha,  # LoRA的alpha参数, scaling=alpha/r
                target_modules=target_modules,  # 需要应用LoRA的模块名称
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                task_type="FEATURE_EXTRACTION",
            )
            get_peft_model(model, config_qkv)
    # 使用eval的情况
    elif args.eval:
        if not args.fulltune:
            config_qkv = LoraConfig(
                r=args.lora_rank,  # LoRA的秩
                lora_alpha=args.lora_alpha,  # LoRA的alpha参数, scaling=alpha/r
                target_modules=target_modules,  # 需要应用LoRA的模块名称
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                task_type="FEATURE_EXTRACTION",
            )

            get_peft_model(model, config_qkv)

        checkpoint = torch.load(args.finetune, map_location="cpu")
        print(f"Load pre-trained checkpoint from: {args.finetune}")
        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f"{msg=}")

    if args.train_step == 1:
        for name, param in model.named_parameters():
            if name.startswith("image_encoder"):
                param.requires_grad = False
            else:
                param.requires_grad = True
    elif args.train_step == 2:
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            elif name.startswith("head"):
                param.requires_grad = True
            else:
                param.requires_grad = False  # ?????
    else:
        raise ValueError("Invalid train_step")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # 当没有设置为fulltune时
    if args.fulltune:
        print("Full finetune model!!!")
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

    print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"actual lr: {args.lr:.2e}")

    print(f"accumulate grad iterations: {args.accum_iter}")
    print(f"effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.fulltune or args.use_llrd:
        # build optimizer with layer-wise lr decay (lrd)
        param_groups = lrd.param_groups_lrd(
            model_without_ddp,
            args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay,
        )
        for group in param_groups:
            group["lr"] = args.lr * group.pop("lr_scale")  # remove lr_scale，and set lr
    else:
        param_groups = model_without_ddp.parameters()

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Parameter group {i}: lr = {param_group['lr']}")

    loss_scaler = NativeScaler()

    # calculate class_weights for cross entropy loss
    class_percentages = dataset_train.class_counts / dataset_train.class_counts.sum()
    class_weights = 1 / torch.FloatTensor(class_percentages).to(device)
    print(f"{class_weights=}")
    print(f"{class_percentages=}")
    if args.use_cross_entropy_weight:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using cross entropy weight: {class_weights}")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print(f"Not using cross entropy weight")

    # take effect when args.resume is not None
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
            (test_output_loss_total, test_output_loss_un, test_output_loss_ce),
        ) = evaluate(
            args=args,
            data_loader=data_loader_test,
            model=model,
            device=device,
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
            (val_output_loss_total, val_output_loss_un, val_output_loss_ce),
        ) = evaluate(
            args=args,
            data_loader=data_loader_val,
            model=model,
            device=device,
            epoch=epoch,
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
            log_writer.add_scalar("perf/val_acc", val_stats["acc"], epoch)
            log_writer.add_scalar("perf/val_auc", val_auc_roc, epoch)
            log_writer.add_scalar("perf/val_loss", val_stats["loss"], epoch)

            # add by xujia
            log_writer.add_scalar("perf/val_total_loss", val_output_loss_total, epoch)
            log_writer.add_scalar("perf/val_loss_un", val_output_loss_un, epoch)
            log_writer.add_scalar("perf/val_loss_ce", val_output_loss_ce, epoch)

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
                f.write(f"{log_stats}\n")

        # early stop code
        if args.early_stop_monitor == "val_auc":
            current_metric = val_auc_roc
        elif args.early_stop_monitor == "val_loss":
            current_metric = -val_output_loss_total

        if args.early_stop_patience > 0:
            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0

            else:
                patience_counter += 1
                print(
                    f"EarlyStopping counter: {patience_counter} out of {args.early_stop_patience}"
                )
                if patience_counter >= args.early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
