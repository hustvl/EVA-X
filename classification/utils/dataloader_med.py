import copy
import os
import random
import numpy as np
import torchvision.transforms.functional
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import torch
import torchvision.transforms as transforms
import pandas as pd
import medpy.io
from torch.utils.data import Dataset


# --------------------------------------------Downstream ChestX-ray14-------------------------------------------
class ChestX_ray14(Dataset):
    def __init__(self, data_dir, file, augment,
                 num_class=14, img_depth=3, heatmap_path=None,
                 data_pct=1, seed=0, mode='train',
                 pretraining=False):
        self.img_list = []
        self.img_label = []

        with open(file, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()
                    imagePath = os.path.join(data_dir, lineItems[0])
                    imageLabel = lineItems[1:num_class + 1]
                    imageLabel = [int(i) for i in imageLabel]
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

        self.augment = augment
        self.img_depth = img_depth
        if heatmap_path is not None:
            # self.heatmap = cv2.imread(heatmap_path)
            self.heatmap = Image.open(heatmap_path).convert('RGB')
        else:
            self.heatmap = None
        self.pretraining = pretraining

        if data_pct < 1.0 and data_pct > 0.0 and mode == 'train':
            # set random seed
            random.seed(seed)
            
            # random choice
            index = random.sample(range(len(self.img_list)), int(len(self.img_list) * data_pct))
            self.img_list = [self.img_list[i] for i in index]
            self.img_label = [self.img_label[i] for i in index]
        
        self.mode = mode
        self.pct = data_pct
        if self.pct < 1 and mode == 'train':
            self.true_len = len(self.img_list)
            self.merge_len = int(self.true_len / self.pct)
        

    def __len__(self):
        if self.pct < 1 and self.mode == 'train':
            return self.merge_len
        return len(self.img_list)

    def __getitem__(self, index):

        if self.pct < 1 and self.mode == 'train':
            index = index % self.true_len

        file = self.img_list[index]
        label = self.img_label[index]

        imageData = Image.open(file).convert('RGB')
        if self.heatmap is None:
            imageData = self.augment(imageData)
            img = imageData
            label = torch.tensor(label, dtype=torch.float)
            if self.pretraining:
                label = -1
            return img, label
        else:
            # heatmap = Image.open('nih_bbox_heatmap.png')
            heatmap = self.heatmap
            # heatmap = torchvision.transforms.functional.to_pil_image(self.heatmap)
            imageData, heatmap = self.augment(imageData, heatmap)
            img = imageData
            # heatmap = torch.tensor(np.array(heatmap), dtype=torch.float)
            heatmap = heatmap.permute(1, 2, 0)
            label = torch.tensor(label, dtype=torch.float)
            if self.pretraining:
                label = -1
            return [img, heatmap], label


class Covidx(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transform, num_classes, data_pct=1, seed=0, rank=0, train_list=None, test_list=None):
        self.data_dir = data_dir
        self.phase = phase

        if num_classes == 3:
            self.classes = ['normal', 'pneumonia', 'COVID-19']
        elif num_classes == 2:
            self.classes = ['negative', 'positive']
        else:
            raise ValueError('Wrong number of classes.')
        
        self.class2label = {c: i for i, c in enumerate(self.classes)}

        # collect training/testing files
        if phase == 'train':
            with open(train_list, 'r') as f:
                lines = f.readlines()
        elif phase == 'test':
            with open(test_list, 'r') as f:
                lines = f.readlines()
        lines = [line.strip() for line in lines]
        self.datalist = list()
        for line in lines:
            patient_id, fname, label, source = line.split(' ')
            if phase in ('train', 'val'):
                train_file_path = os.path.join(data_dir, 'train', fname)
                self.datalist.append((train_file_path, label))
                if not os.path.exists(train_file_path):
                    raise FileNotFoundError(train_file_path)
            else:
                if num_classes == 3:
                    test_file_path = os.path.join(data_dir, 'test', fname)
                elif num_classes == 2:
                    test_file_path = os.path.join(data_dir, 'val', fname)
                self.datalist.append((test_file_path, label))
                if not os.path.exists(test_file_path):
                    raise FileNotFoundError(test_file_path)

        self.transform = transform
        self.phase = phase

        if data_pct < 1.0 and data_pct > 0.0 and phase == 'train':
            # print in red
            print('\033[91m' + 'Using {}% data for training'.format(data_pct * 100) + '\033[0m')
            # set random seed
            random.seed(seed)
            # random choice
            index = random.sample(range(len(self.datalist)), int(len(self.datalist) * data_pct))
            self.datalist = [self.datalist[i] for i in index]

            # # save index with rank for debug
            # with open('index_{}.txt'.format(rank), 'w') as f:
            #     for i in index:
            #         f.write(str(i) + '\n')

            # length
            self.true_len = len(self.datalist)
            self.merge_len = int(self.true_len / data_pct)
        self.pct = data_pct
        self.phase = phase

    def __len__(self):
        if self.phase == 'train' and self.pct < 1:
            return self.merge_len
        return len(self.datalist)

    def __getitem__(self, index):

        if self.phase == 'train' and self.pct < 1:
            index = index % self.true_len

        fpath, label = self.datalist[index]
        image = Image.open(fpath).convert('RGB')
        image = self.transform(image)
        label = self.class2label[label]
        label = torch.tensor(label, dtype=torch.long)
        return image, label

class CheXpert(Dataset):
    '''
    Reference:
        @inproceedings{yuan2021robust,
            title={Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
            author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
            booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
            year={2021}
            }
    '''

    def __init__(self,
                 csv_path,
                 image_root_path='',
                 class_index=0,
                 use_frontal=True,
                 use_upsampling=True,
                 flip_label=False,
                 shuffle=True,
                 seed=123,
                 verbose=True,
                 transform=None,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion'],
                 mode='train',
                 heatmap_path=None,
                 pretraining=False,
                 unique_patients=True,
                 use_rand_label=False
                 ):

        # load data from csv
        self.df = pd.read_csv(csv_path)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '')
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '')
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

            # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print('Upsampling %s...' % col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        if heatmap_path is not None:
            # self.heatmap = cv2.imread(heatmap_path)
            self.heatmap = Image.open(heatmap_path).convert('RGB')

        else:
            self.heatmap = None

        # # Remove rows with -1
        # original_len = len(self.df)
        # self.df = self.df[~(self.df == -1).any(axis=1)]
        # print('Removed %d rows with -1 from %d rows' % (original_len - len(self.df), original_len))

        if mode=='test' and unique_patients:
            self.df["PatientID"] = self.df["Path"].str.extract(pat=r'(patient\d+)')
            self.df = self.df.groupby("PatientID").first().reset_index()

        # impute missing values
        if use_rand_label:
            for col in train_cols:
                if col in ['Edema', 'Atelectasis']:
                    self.df[col].replace(-1, random.uniform(0.7, 0.8), inplace=True)
                    self.df[col].fillna(0, inplace=True)
                elif col in ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']:
                    self.df[col].replace(-1,  random.uniform(0.2, 0.3), inplace=True)
                    self.df[col].fillna(0, inplace=True)
                else:
                    self.df[col].fillna(0, inplace=True)

        else:
            for col in train_cols:
                if col in ['Edema', 'Atelectasis']:
                    self.df[col].replace(-1,  1, inplace=True)
                    self.df[col].fillna(0, inplace=True)
                elif col in ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']:
                    self.df[col].replace(-1,  0, inplace=True)
                    self.df[col].fillna(0, inplace=True)
                else:
                    self.df[col].fillna(0, inplace=True)


        self._num_images = len(self.df)

        # 0 --> -1
        if flip_label and class_index != -1:  # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)

            # shuffle data
        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

        assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        assert image_root_path != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1:  # 5 classes
            print('Multi-label mode: True, Number of classes: [%d]' % len(train_cols))
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:  # 1 class
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()

        self.mode = mode
        self.class_index = class_index

        self.transform = transform

        self._images_list = [image_root_path + path for path in self.df['Path'].tolist()]
        if class_index != -1:
            self._labels_list = self.df[train_cols].values[:, class_index].tolist()
        else:
            self._labels_list = self.df[train_cols].values.tolist()

        if verbose:
            if class_index != -1:
                print('-' * 30)
                if flip_label:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[-1] + self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1]))
                    print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                else:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[0] + self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[0]))
                    print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                print('-' * 30)
            else:
                print('-' * 30)
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    imratio = self.value_counts_dict[class_key][1] / (
                            self.value_counts_dict[class_key][0] + self.value_counts_dict[class_key][1])
                    imratio_list.append(imratio)
                    print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0]))
                    print('%s(C%s): imbalance ratio is %.4f' % (select_col, class_key, imratio))
                    print()
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list
                print('-' * 30)
        self.pretraining = pretraining

    @property
    def class_counts(self):
        return self.value_counts_dict

    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)

    @property
    def data_size(self):
        return self._num_images

    def __len__(self):
        if self.mode == 'train':
            return self._num_images // 20
        return self._num_images

    def __getitem__(self, idx):

        if self.mode == 'train':
            idx = random.randint(0, 9)*self.__len__() + idx

        if self.heatmap is None:
            image = Image.open(self._images_list[idx]).convert('RGB')

            image = self.transform(image)

            # image = image.transpose((2, 0, 1)).astype(np.float32)

            if self.class_index != -1:  # multi-class mode
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)
                # label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
            else:
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)

            if self.pretraining:
                label = -1

            return image, label
        else:
            # heatmap = Image.open('nih_bbox_heatmap.png')
            heatmap = self.heatmap
            image = Image.open(self._images_list[idx]).convert('RGB')
            image, heatmap = self.transform(image, heatmap)
            heatmap = heatmap.permute(1, 2, 0)
            # heatmap = torchvision.transforms.functional.to_pil_image(self.heatmap)
            if self.class_index != -1:  # multi-class mode
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)
                # label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
            else:
                label = torch.tensor(self._labels_list[idx], dtype=torch.float32).reshape(-1)

            if self.pretraining:
                label = -1

            return [image, heatmap], label