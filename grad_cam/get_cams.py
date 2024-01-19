"""
We use the file to generate grad-cam of EVA-X / ViT / DN121 
models on NIH Chest-Xray14 dataset.
Reference:  https://github.com/jacobgil/pytorch-grad-cam
            https://github.com/mlmed/torchxrayvision
Thanks to their work!

Written by Jingfeng Yao, 
from HUST-VL
"""

import os
import cv2
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from utils import load_weights_for_eva
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
# warn without warnings
import warnings
warnings.filterwarnings("ignore")

default_pathologies = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltrate",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural thickening",
    "Hernia"
]

def get_args():
    parser = argparse.ArgumentParser(description='Grad-CAM for EVA-X')
    
    parser.add_argument('--data-root', 
                        type=str, default='/home/jingfengyao/code/medical/EVA_Medical/datasets/cxr14')
    parser.add_argument('--box-csv', 
                        type=str, default='/data/jingfengyao/datasets/NIH/BBox_List_2017.csv')
    parser.add_argument('--model', 
                        type=str, default='resnet50') # eva_x_small / vit_small_patch16 / densenet121 / resnet50
    parser.add_argument('--ckpt', 
                        type=str, default='/home/jingfengyao/code/medical/EVA-X/classification/pretrained/r50_biovil.pth')
    parser.add_argument('--save-path', 
                        type=str, default='./vit_grad_cam')
    parser.add_argument('--thrd', 
                        type=float, default=0.4)
    parser.add_argument('--searching',
                        action='store_true')
    args = parser.parse_args()
    return args

# define dataset
class Chest_Xray14(Dataset):
    def __init__(self,
                 data_root=None,
                 box_csv=None, 
                ):
        super(Chest_Xray14, self).__init__()
        self.data_root = data_root
        self.box_csv = box_csv
        # read csv file
        self.data = pd.read_csv(box_csv)
        mean=[0.5056, 0.5056, 0.5056]
        std=[0.252, 0.252, 0.252]
        self.augment = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=mean,
                                std=std)
        ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # read image
        image_path = self.data.iloc[index]['Image Index']
        image = cv2.imread(os.path.join(self.data_root, image_path))

        # read bbox
        x = int(self.data.iloc[index]['Bbox [x'])
        y = int(self.data.iloc[index]['y'])
        w = int(self.data.iloc[index]['w'])
        h = int(self.data.iloc[index]['h]'])

        # save image with bbox for debug
        if False:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.imwrite('test.jpg', image)

        # rgb image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (224, 224))

        # tensor image
        image = self.augment(image)

        # read label
        label = self.data.iloc[index]['Finding Label']

        # normalize bbox
        x = x / 1024.
        y = y / 1024.
        w = w / 1024.
        h = h / 1024.

        return image, image_rgb, label, x, y, w, h

def reshape_transform_vit_huggingface(x):
    activations=x[:, 1:]
    # Reshape to a 12 x 12 spatial image:
    activations = activations.view(activations.shape[0], 14, 14, activations.shape[2])
    # Transpose the features to be in the second coordinate:
    activations = activations.transpose(2, 3).transpose(1, 2)
    return activations

def main():
    # get args
    args = get_args()

    import datetime

    def print_with_prefix(message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp} EVA-X]: {message}")

    # chest-xray14 dataset
    dataset = Chest_Xray14(args.data_root, args.box_csv)
    dataset[0]

    # get models
    print_with_prefix('Loading model: {}'.format(args.model))
    if args.model == 'eva_x_small':
        import timm
        import model_eva
        sys.stdout = open(os.devnull, 'w')
        model = timm.create_model(
            'eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE',
            pretrained=False,
            img_size=224,
            num_classes=14,
            drop_rate=0,
            drop_path_rate=0.2,
            attn_drop_rate=0,
            drop_block_rate=None,
            use_mean_pooling=True,
        )
        sys.stdout = sys.__stdout__
    elif args.model == 'vit_small_patch16':
        import models_vit
        model = models_vit.__dict__['vit_small_patch16'](
            img_size=224,
            num_classes=14,
            drop_rate=0,
            drop_path_rate=0.2,
            global_pool=True,
        )
    elif args.model == 'densenet121':
        from model_cnns import DenseNet121
        model = DenseNet121(num_classes=14)
    elif args.model == 'resnet50':
        from model_cnns import ResNet50
        model = ResNet50(num_classes=14)
    else:
        raise NotImplementedError
    
    model = model.cuda()
    model.eval()

    # load checkpoint
    print_with_prefix('Loading checkpoint from {}'.format(args.ckpt))
    if args.model == 'eva_x_small':
        checkpoint = torch.load(args.ckpt)
        # Redirect stdout to null device
        sys.stdout = open(os.devnull, 'w')
        msg = load_weights_for_eva(model, checkpoint)
        # Restore stdout
        sys.stdout = sys.__stdout__
        if msg != None:
            raise RuntimeError('load weights failed: {}'.format(msg))
    elif args.model == 'vit_small_patch16':
        checkpoint = torch.load(args.ckpt)
        msg = model.load_state_dict(checkpoint['model'])
    elif args.model == 'densenet121':
        checkpoint = torch.load(args.ckpt)
        msg = model.load_state_dict(checkpoint['model'])
    elif args.model == 'resnet50':
        checkpoint = torch.load(args.ckpt)
        msg = model.load_state_dict(checkpoint['model'])
    else:
        raise NotImplementedError
    print_with_prefix('Loading msg: {}'.format(msg))
    print_with_prefix('Load weights successfully!')

    # get grad-cam
    if args.model == 'eva_x_small' or args.model == 'vit_small_patch16':
        target_layers  = [model.blocks[-1].norm1]
        targets = [BinaryClassifierOutputTarget(1)]
    elif args.model == 'densenet121':
        target_layers = [model.features.denseblock4.denselayer16.norm1]
        targets = [BinaryClassifierOutputTarget(1)]
    elif args.model == 'resnet50':
        target_layers = [model.layer4[2].bn3]
        targets = [BinaryClassifierOutputTarget(1)]

    # make save path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    iou_list = []
    if args.searching:
        results_list = []
        gts = []

    # calculate cam
    print_with_prefix('Calculating grad-cam...')
    for i in tqdm(range(len(dataset))):
        image_tensor, image_rgb, label, x, y, w, h = dataset[i]
        image_tensor = image_tensor.unsqueeze(0).cuda()

        # get specific label
        label_id = default_pathologies.index(label)
        model.set_specific_id(label_id)

        # get grad-cam
        if args.model == 'eva_x_small' or args.model == 'vit_small_patch16':
            cam = GradCAM(model=model, 
                        target_layers=target_layers, 
                        use_cuda=True, 
                        reshape_transform=reshape_transform_vit_huggingface)
        elif args.model == 'densenet121' or args.model == 'resnet50':
            cam = GradCAM(model=model, 
                        target_layers=target_layers, 
                        use_cuda=True)
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        grayscale_cam = 1 - grayscale_cam
        visualization = show_cam_on_image((image_rgb / 255.), grayscale_cam, use_rgb=True)
        # add bbox
        x = int(x * 224)
        y = int(y * 224)
        w = int(w * 224)
        h = int(h * 224)
        cv2.rectangle(visualization, (x, y), (x+w, y+h), (255, 255, 255), 2)
        
        # calulate iou between grad-cam and bbox
        grayscale_cam = 1 - grayscale_cam
        map = grayscale_cam > args.thrd
        map = map.astype(np.uint8)

        bbox = np.zeros((224, 224))
        bbox[y:y+h, x:x+w] = 1

        intersection = np.logical_and(map, bbox)
        union = np.logical_or(map, bbox)
        iou_score = np.sum(intersection) / np.sum(union)

        if not args.searching:
            # write iou in image
            cv2.putText(visualization, 'iou: {:.2f}'.format(iou_score), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # model predict
        if False:
            out = model(image_tensor)
            if out > 0:
                pred = 'pos'
            else:
                pred = 'neg'
            # write pred in image
            cv2.putText(visualization, 'pred: {}'.format(pred), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if not args.searching:
            cv2.imwrite(os.path.join(args.save_path, '{}_{}.jpg'.format(label, i)), visualization)
        
        iou_list.append(iou_score)
        if args.searching:
            results_list.append(grayscale_cam)
            gts.append(bbox)

    if not args.searching:

        # print thrd
        print_with_prefix('thrd: {}'.format(args.thrd))

        # calculate mean iou
        iou_list = np.array(iou_list)
        mean_iou = np.mean(iou_list)
        print_with_prefix('mean iou: {:.4f}'.format(mean_iou))

        # average precision of iou > 0.25
        iou_list_25 = iou_list > 0.25
        iou_list_25 = iou_list_25.astype(np.float32)
        ap = np.mean(iou_list_25)
        print_with_prefix('AP25: {:.4f}'.format(ap))

        # average precision of iou > 0.5
        iou_list_5 = iou_list > 0.5
        iou_list_5 = iou_list_5.astype(np.float32)
        ap = np.mean(iou_list_5)
        print_with_prefix('AP5: {:.4f}'.format(ap))
    
    elif args.searching:
        print_with_prefix('Searching for best thrd...')
        best_mAP = 0
        for i in range(1, 61, 1):
            thrd = i / 100
            iou_list = []
            for j in range(len(results_list)):
                map = results_list[j] > thrd
                map = map.astype(np.uint8)

                bbox = gts[j]

                intersection = np.logical_and(map, bbox)
                union = np.logical_or(map, bbox)
                iou_score = np.sum(intersection) / np.sum(union)

                iou_list.append(iou_score)
            iou_list = np.array(iou_list)
            mean_iou = np.mean(iou_list)

            # AP25
            iou_list_25 = iou_list > 0.25
            iou_list_25 = iou_list_25.astype(np.float32)
            ap25 = np.mean(iou_list_25)

            # AP50
            iou_list_5 = iou_list > 0.5
            iou_list_5 = iou_list_5.astype(np.float32)
            ap50 = np.mean(iou_list_5)

            print_with_prefix('thrd: {:.2f}, mean iou: {:.4f}, AP25: {:.4f}, AP50: {:.4f}, mAP25-50: {:.4f}'.format(thrd, mean_iou, ap25, ap50, (ap25+ap50)/2))

            if (ap25+ap50)/2 > best_mAP:
                best_mAP = (ap25+ap50)/2
                best_thrd = thrd
                best_miou = mean_iou
                best_ap25 = ap25
                best_ap50 = ap50
        print_with_prefix('Best: thrd: {:.2f}, AP25: {:.4f}, AP50: {:.4f}, mean iou: {:.4f},  mAP25-50: {:.4f}'.format(best_thrd, best_ap25, best_ap50, best_miou, best_mAP))
        
if __name__ == '__main__':
    main()