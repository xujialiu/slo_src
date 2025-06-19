from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import io
import torch
from albumentations import ImageOnlyTransform


def build_dataset_in(is_train, args):
    if args.dataset_type == "fp":
        build_transform = build_transform_fp
    elif args.dataset_type == "slo":
        build_transform = build_transform_slo
    else:
        raise ValueError("dataset_type should be 'fp' or'slo'.")

    transform = build_transform(is_train, args)
    dataset = LoadDataIn(
        csv_path=args.csv_path,
        data_path=args.data_path,
        is_train=is_train,
        transform=transform,
    )
    return dataset


def build_dataset_out(is_train, args):
    if args.dataset_type == "fp":
        build_transform = build_transform_fp
    elif args.dataset_type == "slo":
        build_transform = build_transform_slo
    else:
        raise ValueError("dataset_type should be 'fp' or'slo'.")

    transform = build_transform(is_train, args)
    dataset = LoadDataOut(
        csv_path=args.csv_path_out,
        data_path=args.data_path_out,
        is_train=is_train,
        transform=transform,
    )
    return dataset


class LoadDataIn(Dataset):
    def __init__(self, csv_path, data_path, is_train="train", transform=None):
        super().__init__()
        self.csv_path = csv_path
        self.data_path = Path(data_path)
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        if is_train == "train":
            self.df = self.df.loc[self.df.train_test == "train"].reset_index(drop=True)
        elif is_train == "val":
            self.df = self.df.loc[self.df.train_test == "val"].reset_index(drop=True)
        elif is_train == "test":
            self.df = self.df.loc[self.df.train_test == "test"].reset_index(drop=True)
        elif is_train == "all":
            pass
        else:
            raise ValueError("is_train should be 'train', 'val', or 'test'.")
        self.class_counts = self.df.label.value_counts().sort_index().values
        self.labels = self.df.label.values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = self.df.loc[index, "name"]
        img_path = self.data_path / img_name

        img = io.imread(str(img_path))

        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif (img.shape[-1] != 3) or (img.ndim != 3):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = self.transform(image=img)["image"]

        label = self.df.loc[index, "label"]
        return (img, label)


class LoadDataOut(Dataset):
    def __init__(self, csv_path, data_path, is_train="train", transform=None):
        super().__init__()
        self.csv_path = csv_path
        self.data_path = Path(data_path)
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        if is_train == "train":
            self.df = self.df.loc[self.df.train_test == "train"].reset_index(drop=True)
        elif is_train == "val":
            self.df = self.df.loc[self.df.train_test == "val"].reset_index(drop=True)
        elif is_train == "test":
            self.df = self.df.loc[self.df.train_test == "test"].reset_index(drop=True)
        elif is_train == "all":
            pass
        else:
            raise ValueError("is_train should be 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = self.df.loc[index, "name"]
        img_path = self.data_path / img_name

        img = io.imread(str(img_path))

        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif (img.shape[-1] != 3) or (img.ndim != 3):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = self.transform(image=img)["image"]
        return img


def build_transform_slo(is_train, args):
    print(f"{is_train} input size: {args.input_size}")

    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # model input size
    height, width = args.input_size

    # train transform
    if is_train == "train":
        if args.random_crop_perc == 1:
            transform = A.Compose(
                [
                    A.Rotate(
                        interpolation=cv2.INTER_CUBIC,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.7,
                        limit=45,
                    ),
                    A.Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC),
                    A.CenterCrop(height=height, width=width),
                    ChannelBrightnessContrast(
                        channel_idx=1, brightness_limit=0.3, contrast_limit=0.3, p=1
                    ),
                    ChannelBrightnessContrast(
                        channel_idx=2, brightness_limit=0.3, contrast_limit=0.3, p=1
                    ),
                    A.ImageCompression(quality_lower=50, quality_upper=100, p=0.2),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.MedianBlur(p=0.2),
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.2),
                    A.GaussNoise(p=0.2),
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=height // 8,
                        max_width=width // 8,
                        min_holes=1,
                        min_height=height // 32,
                        min_width=width // 32,
                        p=0.5,
                    ),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )
        elif args.random_crop_perc != 1:
            image_height, image_width = (
                int(height // args.random_crop_perc) + 1,
                int(width // args.random_crop_perc) + 1,
            )
            transform = A.Compose(
                [
                    A.Rotate(
                        interpolation=cv2.INTER_CUBIC,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.7,
                        limit=45,
                    ),
                    A.Resize(
                        height=image_height,
                        width=image_width,
                        interpolation=cv2.INTER_CUBIC,
                    ),
                    A.RandomCrop(
                        height=int(image_height * args.random_crop_perc),
                        width=int(image_width * args.random_crop_perc),
                    ),
                    A.Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC),
                    A.CenterCrop(height=height, width=width),
                    ChannelBrightnessContrast(
                        channel_idx=1, brightness_limit=0.3, contrast_limit=0.3, p=1
                    ),
                    ChannelBrightnessContrast(
                        channel_idx=2, brightness_limit=0.3, contrast_limit=0.3, p=1
                    ),
                    A.ImageCompression(quality_lower=50, quality_upper=100, p=0.2),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.MedianBlur(p=0.2),
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.2),
                    A.GaussNoise(p=0.2),
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=height // 8,
                        max_width=width // 8,
                        min_holes=1,
                        min_height=height // 32,
                        min_width=width // 32,
                        p=0.5,
                    ),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )

    else:
        # eval transform
        transform = A.Compose(
            [
                A.Resize(
                    height=height,
                    width=width,
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.CenterCrop(
                    height=int(height * args.random_crop_perc),
                    width=int(width * args.random_crop_perc),
                ),
                A.Resize(
                    height=height,
                    width=width,
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    return transform


def build_transform_fp(is_train, args):
    print(f"{is_train} input size: {args.input_size}")

    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # model input size
    height, width = args.input_size

    # train transform
    if is_train == "train":
        if args.random_crop_perc == 1:
            transform = A.Compose(
                [
                    A.Rotate(
                        interpolation=cv2.INTER_CUBIC,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.7,
                        limit=45,
                    ),
                    A.Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC),
                    A.CenterCrop(height=height, width=width),
                    A.ImageCompression(quality_lower=50, quality_upper=100, p=0.2),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.MedianBlur(p=0.2),
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.2),
                    A.GaussNoise(p=0.2),
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=height // 8,
                        max_width=width // 8,
                        min_holes=1,
                        min_height=height // 32,
                        min_width=width // 32,
                        p=0.5,
                    ),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )
        elif args.random_crop_perc != 1:
            image_height, image_width = (
                int(height // args.random_crop_perc) + 1,
                int(width // args.random_crop_perc) + 1,
            )
            transform = A.Compose(
                [
                    A.Rotate(
                        interpolation=cv2.INTER_CUBIC,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.7,
                        limit=45,
                    ),
                    A.Resize(
                        height=image_height,
                        width=image_width,
                        interpolation=cv2.INTER_CUBIC,
                    ),
                    A.RandomCrop(
                        height=int(image_height * args.random_crop_perc),
                        width=int(image_width * args.random_crop_perc),
                    ),
                    A.Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC),
                    A.CenterCrop(height=height, width=width),
                    A.ImageCompression(quality_lower=50, quality_upper=100, p=0.2),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.MedianBlur(p=0.2),
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.2),
                    A.GaussNoise(p=0.2),
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=height // 8,
                        max_width=width // 8,
                        min_holes=1,
                        min_height=height // 32,
                        min_width=width // 32,
                        p=0.5,
                    ),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )

    else:
        # eval transform
        transform = A.Compose(
            [
                A.Resize(
                    height=height,
                    width=width,
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.CenterCrop(
                    height=int(height * args.random_crop_perc),
                    width=int(width * args.random_crop_perc),
                ),
                A.Resize(
                    height=height,
                    width=width,
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    return transform


def get_weighted_sampler(dataset):
    class_counts = dataset.class_counts

    print(f"Class counts: {class_counts}")

    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for label in dataset.labels]

    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return weighted_sampler


class ChannelBrightnessContrast(ImageOnlyTransform):
    def __init__(
        self,
        channel_idx,
        brightness_limit=0.2,
        contrast_limit=0.2,
        always_apply=False,
        p=0.5,
    ):
        super(ChannelBrightnessContrast, self).__init__(always_apply, p)
        self.channel_idx = channel_idx
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        # 初始化内置的亮度对比度变换
        self.transform = A.RandomBrightnessContrast(
            brightness_limit=self.brightness_limit,
            contrast_limit=self.contrast_limit,
            p=1.0,  # 强制每次调用都应用
        )

    def apply(self, image, **params):
        # 提取目标通道（保持三维形状 HWC）
        channel = image[:, :, self.channel_idx : self.channel_idx + 1]
        # 应用亮度对比度变换
        transformed = self.transform(image=channel)["image"]
        # 将结果覆盖回原图像的对应通道
        image[:, :, self.channel_idx] = transformed[:, :, 0]
        return image
