import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import cv2
from pathlib import Path
from PIL import Image


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    dataset = LoadData(
        csv_path=args.csv_path,
        data_path=args.data_path,
        is_train=is_train,
        transform=transform,
    )
    return dataset


class LoadData(Dataset):
    def __init__(self, csv_path, data_path, is_train=True, transform=None):
        super().__init__()
        self.csv_path = csv_path
        self.data_path = Path(data_path)
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        if is_train == "train":
            self.df = self.df.loc[self.df.train_test == "train"].reset_index(drop=True)
        else:
            self.df = self.df.loc[self.df.train_test == "test"].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = self.df.loc[index, "name"]
        img_eye = self.df.loc[index, "eye"]
        img_path = self.data_path / img_eye / img_name

        img = Image.open(str(img_path))
        img = self.transform(img)

        label = self.df.loc[index, "label"]
        return (img, label)


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train == "train":
        # this should always dispatch to transforms_imagenet_train
        print(args.input_size, type(args.input_size))
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )

    else:
        # eval transform
        t = []
        if args.input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(args.input_size / crop_pct)
        
        t.append(
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        )
        t.append(transforms.CenterCrop(size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        transform = transforms.Compose(t)

    return transform
