import os
import random
import shutil
import pydicom
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

def process_siim(
        folder_path,
        csv_path,
        image_save_path,
        mask_save_path,
        ratio=0.7,
        debug_mode=False,
        train_list_path=None,
        test_list_path=None
        ):

    if debug_mode:
        warnings.warn('Debug mode is on. This will save the combined image and mask in the current directory.')

    def read_files_in_folder(folder_path):
        file_dict = {}
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_name = os.path.basename(file_path[:-4])
                file_dict[file_name] = file_path
        return file_dict

    def read_csv_to_dict(csv_path):
        data_dict = {}
        df = pd.read_csv(csv_path)
        for index, row in df.iterrows():
            image_id = row['ImageId']
            rle_code = row[' EncodedPixels']
            data_dict[image_id] = rle_code
        return data_dict

    def rle2mask(rle, width, height):
        mask = np.zeros(width * height)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position + lengths[index]] = 1
            current_position += lengths[index]

        return mask.reshape(width, height).T

    def split_ids(id_list, ratio=0.7):
        total_ids = len(id_list)
        split_index = int(total_ids * ratio)
        train_ids = id_list[:split_index]
        test_ids = id_list[split_index:]
        return train_ids, test_ids

    file_dict = read_files_in_folder(folder_path)
    data_dict = read_csv_to_dict(csv_path)

    image_with_mask_num=0
    os.makedirs(image_save_path, exist_ok=True)
    os.makedirs(mask_save_path, exist_ok=True)

    for image_id, file_path in tqdm(file_dict.items()):
        rle_mask = data_dict.get(image_id)
        if rle_mask:
            if str(rle_mask) == '-1':
                pass
            else:
                # Read the dcm file and store it in the 'image' variable
                image = pydicom.dcmread(file_path)
                image = image.pixel_array
                mask = rle2mask(rle_mask, image.shape[0], image.shape[1])
                mask = mask.astype(np.uint8)

                # Save the image and mask
                _image_save_path = os.path.join(image_save_path, image_id + '.png')
                _mask_save_path = os.path.join(mask_save_path, image_id + '.png')

                # Save the image as grayscale
                image = image.astype(np.uint8)  # Convert to grayscale (0-255)
                image = Image.fromarray(image)
                mask = Image.fromarray(mask)

                if debug_mode:
                    # Combine the image and mask
                    combined_image = Image.blend(image, mask, alpha=0.2)
                    combined_image.save('combined.png')
                    import ipdb; ipdb.set_trace()

                image.save(_image_save_path)
                mask.save(_mask_save_path)

                image_with_mask_num = image_with_mask_num + 1

    print('Total number of images with mask: {}'.format(image_with_mask_num))

    id_list = [x[:-4] for x in os.listdir(image_save_path) if x.endswith('.png') or x.endswith('.jpg')]

    os.makedirs(os.path.join(image_save_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(image_save_path, 'test'), exist_ok=True)
    os.makedirs(os.path.join(mask_save_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(mask_save_path, 'test'), exist_ok=True)

    try:
        train_ids = [x.strip() for x in open(train_list_path, 'r').readlines()]
        test_ids = [x.strip() for x in open(test_list_path, 'r').readlines()]
    except:
        raise FileNotFoundError(f"Train_file: {train_list_path} or test_file: {test_list_path} don't exist, check the split file.")
        # # print in red
        # print('\033[91m' + 'Warning: No train.txt and test.txt found. Using random split instead.' + '\033[0m')
        # train_ids, test_ids = split_ids(id_list, ratio)
    print('Total number of images: {}'.format(len(id_list)))

    print('Moving images to train folders...')
    for image_id in tqdm(train_ids):
        image_path = os.path.join(image_save_path, image_id + '.png')
        mask_path = os.path.join(mask_save_path, image_id + '.png')
        shutil.copy(image_path, os.path.join(image_save_path, 'train'))
        shutil.copy(mask_path, os.path.join(mask_save_path, 'train'))

    print('Moving images to test folders...')
    for image_id in tqdm(test_ids):
        image_path = os.path.join(image_save_path, image_id + '.png')
        mask_path = os.path.join(mask_save_path, image_id + '.png')
        shutil.copy(image_path, os.path.join(image_save_path, 'test'))
        shutil.copy(mask_path, os.path.join(mask_save_path, 'test'))

    print('Done!')

if __name__ == '__main__':
    folder_path = '/data/jingfengyao/datasets/siim_stage2/SIIM_Pneumothorax/dicom-images-train'
    csv_path = '/data/jingfengyao/datasets/siim_stage2/SIIM_Pneumothorax/train-rle.csv'
    image_save_path = 'dataset/SIIM_Pneumothorax/images'
    mask_save_path = 'dataset/SIIM_Pneumothorax/masks'
    train_list_path = 'data_splits/siim/train.txt'
    test_list_path = 'data_splits/siim/test.txt'

    process_siim(folder_path, csv_path, image_save_path, mask_save_path, 0.7, False, train_list_path, test_list_path)
