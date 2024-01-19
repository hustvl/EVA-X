import os
import cv2
import numpy as np
import random
import shutil

def merge_masks(folder_path, save_root):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # make the save folder if it doesn't exist
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # Filter the list to include only PNG files
    png_files = [file for file in file_list if file.endswith('.png')]

    # Create an empty array to store the merged mask
    merged_mask = None

    # Get the unique prefixes from the file names
    prefixes = set([file[:13] for file in png_files])

    # Iterate over the prefixes
    for prefix in prefixes:
        # Get the files with the current prefix
        prefix_files = [file for file in png_files if file.startswith(prefix)]

        # Iterate over the prefix files
        for file in prefix_files:
            # Read the image
            image_path = os.path.join(folder_path, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # If this is the first image, initialize the merged mask
            if merged_mask is None:
                merged_mask = np.zeros_like(image)

            # Add the image to the merged mask
            merged_mask = np.maximum(merged_mask, image)

            # 255 -> 1
            merged_mask[merged_mask == 255] = 1

        # Save the merged mask with the prefix as the file name
        save_path = os.path.join(save_root, prefix + '.png')
        cv2.imwrite(save_path, merged_mask)
        
        merged_mask=None

def split_dataset(image_folder, mask_folder, save_folder, train_split=None, test_split=None):
    # Create the save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Create the train and test folders if they don't exist
    train_folder = os.path.join(save_folder, 'train')
    test_folder = os.path.join(save_folder, 'test')
    train_mask_folder = os.path.join(save_folder, 'train_masks')
    test_mask_folder = os.path.join(save_folder, 'test_masks')
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    if not os.path.exists(train_mask_folder):
        os.makedirs(train_mask_folder)
    if not os.path.exists(test_mask_folder):
        os.makedirs(test_mask_folder)

    # Get a list of all image files in the folder
    image_files = os.listdir(mask_folder)
    image_files = [file for file in image_files if file.endswith('.png')]

    if train_split is None and test_split is None:
        # Shuffle the image files
        random.shuffle(image_files)

        # Calculate the number of images for training and testing
        num_images = len(image_files)
        num_train = int(num_images * 0.7)
        num_test = num_images - num_train

        # Split the image files into training and testing sets
        train_files = image_files[:num_train]
        test_files = image_files[num_train:]
    else:
        # Read the train split file
        with open(train_split, 'r') as f:
            train_files = f.read().splitlines()

        # Read the test split file
        with open(test_split, 'r') as f:
            test_files = f.read().splitlines()

    # Copy the training files to the save folder
    for file in train_files:
        if not file.endswith('.png'):
            file += '.png'
        image_path = os.path.join(image_folder, file)
        mask_path = os.path.join(mask_folder, file)
        save_image_path = os.path.join(save_folder, 'train', file)
        save_mask_path = os.path.join(save_folder, 'train_masks', file)
        shutil.copy(image_path, save_image_path)
        shutil.copy(mask_path, save_mask_path)

    # Copy the testing files to the save folder
    for file in test_files:
        if not file.endswith('.png'):
            file += '.png'
        image_path = os.path.join(image_folder, file)
        mask_path = os.path.join(mask_folder, file)
        save_image_path = os.path.join(save_folder, 'test', file)
        save_mask_path = os.path.join(save_folder, 'test_masks', file)
        shutil.copy(image_path, save_image_path)
        shutil.copy(mask_path, save_mask_path)


if __name__ == '__main__':
    # Define the folder path
    image_folder = '/home/jingfengyao/code/medical/mmsegmentation/projects/unet_cxr/data/jinfeng2/CXR_png'
    folder_path = '/home/jingfengyao/code/medical/mmsegmentation/projects/unet_cxr/data/jinfeng2/Annotations/masks'
    save_folder = '/home/jingfengyao/code/medical/pre-release/segmentation/dataset/Shenzhen'

    # Merge the masks
    merged_mask = merge_masks(folder_path, os.path.join(save_folder, 'merged_masks'))

    # Define the image and mask folder paths
    mask_folder = os.path.join(save_folder, 'merged_masks')
    image_folder = image_folder
    save_folder = save_folder
    split_dataset(image_folder, mask_folder, save_folder)

    # delete the merged mask folder
    shutil.rmtree(os.path.join(save_folder, 'merged_masks'))