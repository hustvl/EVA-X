import os
import re
import time
import json
import torch
import argparse
import datetime
import numpy as np
from pathlib import Path
from libauc import losses
from torchvision import models
from collections import OrderedDict
import torch.backends.cudnn as cudnn
from apex.optimizers import FusedAdam
from torch.utils.tensorboard import SummaryWriter
from timm.models import create_model
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
import timm.optim.optim_factory as optim_factory
import utils.lr_decay as lrd
import utils.misc as misc
from utils.lars import LARS
from utils import  build_dataset_chest_xray, \
                  interpolate_pos_embed, RASampler, NativeScaler
from models import models_eva, models_vit
from engines import train_one_epoch, evaluate_chestxray

def get_args_parser():
    parser = argparse.ArgumentParser('EVA-X fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--use_checkpoint', action='store_true', default=False)
    parser.add_argument('--freeze_backbone', action='store_true', default=False)
    parser.add_argument('--partial_freeze', default=0, type=int)
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # linear probe
    parser.add_argument('--linear_probe', action='store_true')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--stop_grad_conv1', action='store_true')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--train_list", default=None, type=str, help="file for train list")
    parser.add_argument("--val_list", default=None, type=str, help="file for val list")
    parser.add_argument("--test_list", default=None, type=str, help="file for test list")
    parser.add_argument('--eval_interval', default=10, type=int)
    parser.add_argument('--fixed_lr', action='store_true', default=False)
    parser.add_argument('--vit_dropout_rate', type=float, default=0,
                        help='Dropout rate for ViT blocks (default: 0.0)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.add_argument("--build_timm_transform", action='store_true', default=False)
    parser.add_argument("--aug_strategy", default='default', type=str, help="strategy for data augmentation")
    parser.add_argument("--dataset", default='chestxray', type=str)

    parser.add_argument('--repeated-aug', action='store_true', default=False)

    parser.add_argument("--optimizer", default='adamw', type=str)

    parser.add_argument('--ThreeAugment', action='store_true')  # 3augment

    parser.add_argument('--src', action='store_true')  # simple random crop

    parser.add_argument('--loss_func', default=None, type=str)

    parser.add_argument("--norm_stats", default=None, type=str)

    parser.add_argument("--checkpoint_type", default=None, type=str)

    parser.add_argument("--data_pct", default=1.0, type=float)

    # use_smooth_label
    parser.add_argument('--use_smooth_label', action='store_true', default=False)
    # find unused parameters
    parser.add_argument('--find_unused_parameters', action='store_true', default=False)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))


    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True

    dataset_train = build_dataset_chest_xray(split='train', args=args)
    dataset_test = build_dataset_chest_xray(split='test', args=args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if args.repeated_aug:
        sampler_train = RASampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_test) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    if 'vit' in args.model:
        model = models_vit.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_rate=args.vit_dropout_rate,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    elif 'resnet' in args.model:
        model = models.__dict__[args.model](pretrained=False)
        # Modify the number of classes
        num_classes = args.nb_classes  # Replace with the desired number of classes
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif 'densenet' in args.model or 'resnet' in args.model:
        model = models.__dict__[args.model](num_classes=args.nb_classes)
    elif 'eva' in args.model:
        model = create_model(
            args.model,
            pretrained=False,
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_rate=args.vit_dropout_rate,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            use_checkpoint=args.use_checkpoint,
            stop_grad_conv1=args.stop_grad_conv1,
        )
    else:
        raise NotImplementedError

    # Save args as a dictionary in save_dir
    save_dir = args.output_dir + '/args.json'
    args_dict = vars(args)
    with open(save_dir, 'w') as fp:
        json.dump(args_dict, fp, indent=4)


    if args.finetune and not args.eval:
        if 'vit' in args.model:
            checkpoint = torch.load(args.finetune, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % args.finetune)
            if 'mgca' in args.finetune:
                checkpoint_model = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for k, v in checkpoint_model.items():
                    if 'img_encoder_q.model.' in k:
                        new_state_dict[k.replace('img_encoder_q.model.', '')] = v
                checkpoint_model = new_state_dict
            else:
                checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            if args.global_pool:
                for k in ['fc_norm.weight', 'fc_norm.bias']:
                    try:
                        del checkpoint_model[k]
                    except:
                        pass


            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)


            trunc_normal_(model.head.weight, std=2e-5)
        elif 'densenet' in args.model or 'resnet' in args.model:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % args.finetune)
            if 'state_dict' in checkpoint.keys():
                checkpoint_model = checkpoint['state_dict']
            elif 'model' in checkpoint.keys():
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            if args.checkpoint_type == 'smp_encoder':
                state_dict = checkpoint_model

                new_state_dict = OrderedDict()

                for key, value in state_dict.items():
                    if 'model.encoder.' in key:
                        new_key = key.replace('model.encoder.', '')
                        new_state_dict[new_key] = value
                checkpoint_model = new_state_dict
            elif args.checkpoint_type == 'torchvision_dn121':
                new_state_dict = OrderedDict()
                for key, value in checkpoint_model.items():
                    if 'norm' in key:
                        new_key = re.sub(r'norm.(\d+)', r'norm\1', key)
                        new_state_dict[new_key] = value
                        # print(key, 'to', new_key)
                    elif 'conv' in key:
                        new_key = re.sub(r'conv.(\d+)', r'conv\1', key)
                        new_state_dict[new_key] = value
                        # print(key, 'to', new_key)
                checkpoint_model = new_state_dict
            if "fc.weight" in checkpoint_model.keys():
                checkpoint_model.pop('fc.weight')
            if "fc.bias" in checkpoint_model.keys():
                checkpoint_model.pop('fc.bias')
            if "classifier.weight" in checkpoint_model.keys():
                checkpoint_model.pop("classifier.weight")
            if "classifier.bias" in checkpoint_model.keys():
                checkpoint_model.pop("classifier.bias")
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)
        # initialize eva model (advanced vits)
        elif 'eva' in args.model:
            if args.finetune.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.finetune, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.finetune, map_location='cpu')
            from utils import load_weights_for_eva
            model = load_weights_for_eva(model, checkpoint, args)

    if args.freeze_backbone or args.linear_probe:
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True
        if args.freeze_backbone:
            try:
                for _, p in model.fc_norm.named_parameters():
                    p.requires_grad = True
            except:
                print('no fc_norm, pass')
                pass

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of learnable params (M): %.2f' % (n_parameters / 1.e6))
    if args.linear_probe:
        # print list of learnable params
        for name, param in model_without_ddp.named_parameters():
            if param.requires_grad:
                print(name)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    args.lr = args.blr
    print("lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_parameters)
        model_without_ddp = model.module

    if 'vit' in args.model or 'eva' in args.model:
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay
        )
    else:
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    elif args.optimizer == 'fusedlamb':
        optimizer = FusedAdam(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if args.linear_probe:
        optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print("optimizer = %s" % str(optimizer))

    if args.dataset == 'chestxray':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.dataset == 'covidx':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.dataset == 'chexpert':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError


    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        model.module.load_state_dict(torch.load(args.finetune)['model'])
        test_stats = evaluate_chestxray(data_loader_test, model, device, args)
        print(f"Average AUC of the network on the test set images: {test_stats['auc_avg']:.4f}")
        if args.dataset == 'covidx':
            print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_auc = 0.0

    # Write experiment start time to log
    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write("Experiment start time: {}\n".format(datetime.datetime.now()))
    # Write seed to log
    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write("Seed: {}\n".format(seed))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % args.eval_interval == 0 or epoch + 1 == args.epochs):
            if args.dataset != 'chexpert':
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
            else:
                if epoch // 10 == 0:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch)

            test_stats = evaluate_chestxray(data_loader_test, model, device, args)

            # save best model
            if test_stats['auc_avg'] >= max_auc:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch='best')

            print(f"Average AUC on the test set images: {test_stats['auc_avg']:.4f}")
            max_auc = max(max_auc, test_stats['auc_avg'])
            print(f'Max Average AUC: {max_auc:.4f}', {max_auc})

            if args.dataset == 'covidx':
                print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
                max_accuracy = max(max_accuracy, test_stats["acc1"])
                print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.add_scalar('perf/auc_avg', test_stats['auc_avg'], epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

            if args.output_dir and misc.is_main_process() and epoch + 1 == args.epochs:
                # save max auc
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write('max_auc='+str(max_auc) + "\n")
                if args.dataset == 'covidx':
                    # save max accuracy
                    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write('max_accuracy='+str(max_accuracy) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
