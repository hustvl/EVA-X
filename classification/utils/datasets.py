# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from utils.dataloader_med import ChestX_ray14, Covidx, CheXpert

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_dataset_chest_xray(split, args):
    is_train = (split == 'train')
    transform = build_transform(is_train, args)
    if args.dataset == 'chestxray':
        data_list = getattr(args, f'{split}_list')
        dataset = ChestX_ray14(args.data_path, data_list, augment=transform, num_class=14, data_pct=args.data_pct, seed=args.seed, mode='train' if is_train else 'test')
    elif args.dataset == 'covidx':
        print(args.dataset)
        dataset = Covidx(data_dir=args.data_path, phase=split, transform=transform, num_classes=args.nb_classes, data_pct=args.data_pct, seed=args.seed, rank=args.rank, train_list=args.train_list, test_list=args.test_list)
    elif args.dataset == 'chexpert':
        if split == 'train':
            mode = 'train'
        else:
            mode = 'valid'
        data_list = getattr(args, f'{split}_list')
        dataset = CheXpert(csv_path=data_list, image_root_path=args.data_path, use_upsampling=False,
                             use_frontal=True, mode=mode, class_index=-1, transform=transform, use_rand_label=args.use_smooth_label)
    else:
        raise NotImplementedError
    print(dataset)
    if is_train:
        print('train transform: ', transform)
    else:
        print('test transform: ', transform)

    return dataset

def build_transform(is_train, args):

    if 'eva' in args.model:
            mean=(0.49185243, 0.49185243, 0.49185243)
            std=(0.28509309, 0.28509309, 0.28509309)
    else:
        mean = (0.5056, 0.5056, 0.5056)
        std = (0.252, 0.252, 0.252)


    # train transform
    if args.build_timm_transform and is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    if is_train:
        print('\033[91m' + 'The timm transform is NOT activated. Please make sure you meant it.' + '\033[0m')

    if args.dataset != 'chexpert':
        t = []
        if args.input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(args.input_size / crop_pct)
        t.append(
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))

    else:
        if 'tiny' in args.model:
            t = []
            crop_pct = 1.0
            size = int(args.input_size / crop_pct)
            t.append(
                transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(args.input_size))
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
        else:
            # random resize and crop
            t = []
            if is_train:
                if args.input_size <= 224:
                    crop_pct = 224 / 256
                else:
                    crop_pct = 1.0
                size = int(args.input_size / crop_pct)
                t.append(
                    transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms.CenterCrop(args.input_size))
            else:
                t.append(
                    transforms.Resize(int(256), interpolation=PIL.Image.BICUBIC),
                )
                t.append(transforms.CenterCrop(args.input_size))
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))

    return transforms.Compose(t)

