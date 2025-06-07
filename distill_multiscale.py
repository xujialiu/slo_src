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

from timm.models.layers import drop, trunc_normal_
from timm.data.mixup import Mixup
from peft import LoraConfig, get_peft_model

# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay_multiscale as lrd
import util.misc as misc
from util.datasets_distill import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# from engine_finetune import train, evaluate
from engine_distill import train, evaluate

# import vit_model_patches
from model import MultiScale, VisionTransformer

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
        default=f"SLO",
        help="path where results will be saved",
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
    parser.add_argument(
        "--fusion_mlp_ratio",
        default=1,
        type=int,
        help="mlp ratio of fusion layers (default: 1)",
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

    dataset_train = build_dataset(is_train="train", args=args)

    if args.train_all:
        dataset_val = build_dataset(is_train="train", args=args)
    else:
        dataset_val = build_dataset(is_train="val", args=args)

    if args.eval:
        dataset_test = build_dataset(is_train="test", args=args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # create train sampler
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
    )

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
            fusion_num_heads=16,
        )
        teacher_kwargs = EasyDict(
            embed_dim=1024,
            depth=24,
            num_heads=16,
        )
    elif "base" in args.model:
        kwargs = EasyDict(
            embed_dim=768,
            depth=12,
            num_heads=12,
            fusion_num_heads=12,
        )
        teacher_kwargs = EasyDict(
            embed_dim=768,
            depth=12,
            num_heads=12,
        )

    model = MultiScale(
        input_size=args.input_size,
        num_patches=args.num_patches,
        num_classes=0,
        drop_path_rate=args.drop_path,
        fusion_layer_num=args.fusion_layer_num,
        fusion_dropout=args.fusion_dropout,
        fusion_mlp_ratio=args.fusion_mlp_ratio,
        fm_input_size=args.fm_input_size,
        use_learnable_pos_embed=args.use_learnable_pos_embed,
        **kwargs,
    )

    teacher_model = VisionTransformer(
        img_size=args.fm_input_size, num_classes=0, **teacher_kwargs
    )

    teacher_model.load_state_dict(
        torch.load(args.finetune, map_location="cpu")["model"],
        strict=False,
    )
    teacher_model.to(device)
    teacher_model.eval()

    # load pre-trained model
    if not args.finetune:
        raise ValueError("Please specify a pre-trained model for finetuning")

    elif args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")

        print(f"Load pre-trained checkpoint from: {args.resume}")
        checkpoint_model = checkpoint["model"]
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f"{msg=}")

        # manually initialize fc layer
        # trunc_normal_(model.head.weight, std=2e-5)

    elif args.finetune:
        checkpoint = torch.load(args.finetune, map_location="cpu")

        print(f"Load pre-trained checkpoint from: {args.finetune}")
        checkpoint_model = checkpoint["model"]
        checkpoint_model = {
            "image_encoder." + key: value for key, value in checkpoint_model.items()
        }

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f"{msg=}")

        # manually initialize fc layer
        # trunc_normal_(model.head.weight, std=2e-5)

    # freeze the pretrained weights
    for name, param in model.named_parameters():
        if name.startswith("image_encoder"):
            param.requires_grad = False
        elif name.startswith("fusion_block"):
            param.requires_grad = True

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

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

    # if args.fulltune or args.use_llrd:
    #     # build optimizer with layer-wise lr decay (lrd)
    #     param_groups = lrd.param_groups_lrd(
    #         model_without_ddp,
    #         args.weight_decay,
    #         no_weight_decay_list=model_without_ddp.no_weight_decay(),
    #         layer_decay=args.layer_decay,
    #     )
    #     for group in param_groups:
    #         group["lr"] = args.lr * group.pop("lr_scale")  # remove lr_scale，and set lr
    # else:
    #     param_groups = model_without_ddp.parameters()

    param_groups = model_without_ddp.parameters()

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Parameter group {i}: lr = {param_group['lr']}")

    loss_scaler = NativeScaler()

    criterion = torch.nn.MSELoss()

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train(
            model,
            teacher_model,
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

        (val_stats, val_output_loss_total) = evaluate(
            args=args,
            data_loader=data_loader_val,
            model=model,
            teacher_model=teacher_model,
            criterion=criterion,
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
        if min_loss > val_output_loss_total:
            min_loss = val_output_loss_total

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
            log_writer.add_scalar("perf/val_loss", val_stats["loss"], epoch)

            # add by xujia
            log_writer.add_scalar("perf/val_total_loss", val_output_loss_total, epoch)

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
        if args.early_stop_monitor == "val_loss":
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
