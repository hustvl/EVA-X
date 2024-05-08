import os
import cv2
import pydicom
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
from PIL import ImageDraw, ImageFont

def convert_dcm_to_jpg(source_folder, img_folder, mask_folder, csv_file='/data/jingfengyao/datasets/rsna_pneumonia/stage_2_train_labels.csv',
                       train_file='data_splits/rsna/train.txt', test_file='data_splits/rsna/test.txt'):
    # Create the destination folders if they don't exist
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)

    # Create the train and test folders if they don't exist
    if not os.path.exists(os.path.join(img_folder, 'train')):
        os.makedirs(os.path.join(img_folder, 'train'))
    if not os.path.exists(os.path.join(img_folder, 'test')):
        os.makedirs(os.path.join(img_folder, 'test'))
    if not os.path.exists(os.path.join(mask_folder, 'train')):
        os.makedirs(os.path.join(mask_folder, 'train'))
    if not os.path.exists(os.path.join(mask_folder, 'test')):
        os.makedirs(os.path.join(mask_folder, 'test'))

    # Get a list of all DICOM files in the source folder
    dcm_files = [file for file in os.listdir(source_folder) if file.endswith('.dcm')]

    def read_csv(csv_file):
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Extract the relevant columns
        patient_ids = df['patientId'].tolist()
        x_values = df['x'].tolist()
        y_values = df['y'].tolist()
        width_values = df['width'].tolist()
        height_values = df['height'].tolist()
        target_values = df['Target'].tolist()

        # Create a dictionary to store the information
        patient_info = {}
        for i in range(len(patient_ids)):
            patient_id = patient_ids[i]
            x = x_values[i]
            y = y_values[i]
            width = width_values[i]
            height = height_values[i]
            target = target_values[i]
            patient_info[patient_id] = {'x': x, 'y': y, 'width': width, 'height': height, 'target': target}

        return patient_info

    patient_info = read_csv(csv_file)

    # Read the train.txt and test.txt files
    try:
        with open(train_file, 'r') as f:
            train_ids = f.read().splitlines()
        with open(test_file, 'r') as f:
            test_ids = f.read().splitlines()
    except:
        raise FileNotFoundError(f"Train_file: {train_file} or test_file: {test_file} don't exist, check the split file.")
        # # print in red
        # print('\033[91m' + 'Error: train.txt or test.txt not found!' + '\033[0m')

        # # randomly split the data into train and test
        # all_ids = [file.replace('.dcm', '') for file in dcm_files]
        # np.random.shuffle(all_ids)
        # train_ids = all_ids[:int(0.7 * len(all_ids))]
        # test_ids = all_ids[int(0.7 * len(all_ids)):]

    for i, dcm_file in tqdm(enumerate(dcm_files)):
        # Read the DICOM file
        dcm_path = os.path.join(source_folder, dcm_file)
        dcm_data = pydicom.dcmread(dcm_path)

        # Get the image part of the DICOM file
        image = dcm_data.pixel_array

        # Get the relevant information from the CSV file
        patient_id = dcm_data.PatientID
        x_min = patient_info[patient_id]['x']
        y_min = patient_info[patient_id]['y']

        w = patient_info[patient_id]['width']
        h = patient_info[patient_id]['height']
        target = patient_info[patient_id]['target']

        # Check if x and y are not NaN
        if not np.isnan(x_min) and not np.isnan(y_min):
            # Create a mask image
            mask = np.zeros_like(image)

            # Draw the rectangle on the mask image
            cv2.rectangle(mask, (int(x_min), int(y_min)), (int(x_min + w), int(y_min + h)), (255, 255, 255), -1)

            # save image as RGB and mask as grayscale
            img = Image.fromarray(image).convert('RGB')
            msk = Image.fromarray(mask)
            msk = np.array(msk) / 255
            msk = Image.fromarray(msk.astype(np.uint8)).convert('L')

            # Determine if the image should be saved in the train or test folder
            if patient_id in train_ids:
                img.save(os.path.join(img_folder, 'train', dcm_file.replace('.dcm', '.jpg')))
                msk.save(os.path.join(mask_folder, 'train', dcm_file.replace('.dcm', '.png')))
            elif patient_id in test_ids:
                img.save(os.path.join(img_folder, 'test', dcm_file.replace('.dcm', '.jpg')))
                msk.save(os.path.join(mask_folder, 'test', dcm_file.replace('.dcm', '.png')))
            else:
                raise ValueError('Patient ID not found in train or test set!')

    print('Conversion complete!')



# Example usage

csv_file='/data/jingfengyao/datasets/rsna_pneumonia/stage_2_train_labels.csv'
source_folder = '/data/jingfengyao/datasets/rsna_pneumonia/stage_2_train_images'
img_folder = 'dataset/RSNA/images'
mask_folder = 'dataset/RSNA/masks'
convert_dcm_to_jpg(source_folder, img_folder, mask_folder, csv_file)
