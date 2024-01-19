"""
We use the file to prepare the dataset for training and testing.
Modify the path_dict to the path of the unzipped dataset.
"""

import os
import shutil
import datetime
from tqdm import tqdm

def print_with_prefix(msg):
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prefix = f'[{time} EVA-X]: '
    print(f'{prefix}{msg}')

def prepare_cxr14(source_dir, target_dir):

    print_with_prefix('Processing Chest X-Ray14.')

    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        print_with_prefix('Chest X-Ray14 has already been processed, skipping.')
        return
    
    # Iterate through each folder from image_001 to image_012
    for i in range(1, 13):
        print_with_prefix(f'CXR14-Processing image_{str(i).zfill(3)}.')
        folder_name = f'images_{str(i).zfill(3)}'
        folder_path = os.path.join(source_dir, folder_name, 'images')

        # Move all images from the folder to the target directory
        for filename in os.listdir(folder_path):
            source_path = os.path.join(folder_path, filename)
            target_path = os.path.join(target_dir, filename)
            shutil.copy(source_path, target_path)


def prepare_covidx3(source_dir, target_dir):

    print_with_prefix('Processing CovidX-CXR-3.')

    train_dir = os.path.join(source_dir, 'train')
    test_dir = os.path.join(source_dir, 'test')

    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        print_with_prefix('CovidX-CXR-3 has already been processed, skipping.')
        return

    # Copy the train directory to the target directory
    print_with_prefix('CovidX-CXR-3 Copying train. It will take for a while.')
    shutil.copytree(train_dir, os.path.join(target_dir, 'train'))

    # Copy the test directory to the target directory
    print_with_prefix('CovidX-CXR-3-Copying test.')
    shutil.copytree(test_dir, os.path.join(target_dir, 'test'))

def prepare_covidx4(source_dir, target_dir):

    print_with_prefix('Processing CovidX-CXR-4.')

    train_dir = os.path.join(source_dir, 'train')
    test_dir = os.path.join(source_dir, 'val')

    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        print_with_prefix('CovidX-CXR-4 has already been processed, skipping.')
        return

    # Copy the train directory to the target directory
    print_with_prefix('CovidX-CXR-4 Copying train. It will take for a while.')
    shutil.copytree(train_dir, os.path.join(target_dir, 'train'))

    # Copy the test directory to the target directory
    print_with_prefix('CovidX-CXR-4 Copying test.')
    shutil.copytree(test_dir, os.path.join(target_dir, 'val'))

    root_dir = target_dir.split('/')[0]
    # mkdir data_splits
    os.makedirs(os.path.join(root_dir, 'data_splits/covidx4'))

    # Copy train.txt to the target directory
    shutil.copy(os.path.join(source_dir, 'train.txt'), os.path.join(root_dir, 'data_splits/covidx4', 'train.txt'))

    # Copy test.txt to the target directory
    shutil.copy(os.path.join(source_dir, 'val.txt'), os.path.join(root_dir, 'data_splits/covidx4', 'val.txt'))
        
def prepare_chexpert(source_dir, target_dir):
    print_with_prefix('Processing CheXpert.')
    
    # pass if the directory already exists
    if os.path.exists(target_dir):
        print_with_prefix('CheXpert has already been processed, skipping.')
        return
    
    # get root_dir from target_dir
    root_dir = target_dir.split('/')[0]
    # mkdir data_splits
    os.makedirs(os.path.join(root_dir, 'data_splits/chexpert'))

    print_with_prefix('CheXpert Copying images. It will take for a while.')
    shutil.copytree(source_dir, target_dir)

    # move train.csv, valid.csv, test.csv to data_splits/chexpert
    shutil.copy(os.path.join(target_dir, 'train.csv'), os.path.join(root_dir, 'data_splits/chexpert', 'train.csv'))
    shutil.copy(os.path.join(target_dir, 'valid.csv'), os.path.join(root_dir, 'data_splits/chexpert', 'valid.csv'))

def check_dataset(target_dir):

    correct_num = {
        'cxr14': 112120,
        'covidx3_train': 30483,
        'covidx3_test': 400,
        'covidx4_train': 67863,
        'covidx4_test': 8473,
        'chexpert_train': 223415,
        'chexpert_test': 234,
    }

    cxr14_dir = os.path.join(target_dir, 'cxr14')

    if not os.path.exists(cxr14_dir):
        print_with_prefix(f"The 'cxr14' directory does not exist in {target_dir}.")
    else:
        file_count = len(os.listdir(cxr14_dir))
        if file_count != correct_num['cxr14']:
            raise ValueError(f"The 'cxr14' directory exists in {target_dir} with {file_count} files, but it should have {correct_num['cxr14']} files.")
        else:
            print_with_prefix(f"Correct: The 'cxr14' directory exists in {target_dir} with {file_count} files.")

    if not os.path.exists(os.path.join(target_dir, 'covidx3')):
        print_with_prefix(f"The 'covidx3' directory does not exist in {target_dir}.")
    else:
        train_count = len(os.listdir(os.path.join(target_dir, 'covidx3/train')))
        test_count = len(os.listdir(os.path.join(target_dir, 'covidx3/test')))
        if train_count != correct_num['covidx3_train'] or test_count != correct_num['covidx3_test']:
            raise ValueError(f"The 'covidx3' directory exists in {target_dir} with {train_count} train files and {test_count} test files, \
                             but it should have {correct_num['covidx3_train']} train files and {correct_num['covidx3_test']} test files.")
        else:
            print_with_prefix(("Correct: The 'covidx3' directory exists in "
                            f"{target_dir} with {train_count} train files and "
                            f"{test_count} test files."))

    if not os.path.exists(os.path.join(target_dir, 'covidx4')):
        print_with_prefix(f"The 'covidx4' directory does not exist in {target_dir}.")
    else:
        train_count = len(os.listdir(os.path.join(target_dir, 'covidx4/train')))
        test_count = len(os.listdir(os.path.join(target_dir, 'covidx4/val')))

        if train_count != correct_num['covidx4_train'] or test_count != correct_num['covidx4_test']:
            raise ValueError(f"The 'covidx4' directory exists in {target_dir} with {train_count} train files and {test_count} test files, \
                                but it should have {correct_num['covidx4_train']} train files and {correct_num['covidx4_test']} test files.")
        else:
            print_with_prefix(("Correct: The 'covidx4' directory exists in "
                            f"{target_dir} with {train_count} train files and "
                            f"{test_count} test files."))
    
    if not os.path.exists(os.path.join(target_dir, 'chexpert')):
        print_with_prefix(f"The 'chexpert' directory does not exist in {target_dir}.")
    else:
        train_count = 0
        test_count = 0
        for root, dirs, files in os.walk(os.path.join(target_dir, 'chexpert')):
            for file in files:
                if file.endswith('.jpg'):
                    if 'train' in root:
                        train_count += 1
                    elif 'valid' in root:
                        test_count += 1
        if train_count != correct_num['chexpert_train'] or test_count != correct_num['chexpert_test']:
            raise ValueError(f"The 'chexpert' directory exists in {target_dir} with {train_count} train files and {test_count} test files, \
                                but it should have {correct_num['chexpert_train']} train files and {correct_num['chexpert_test']} test files.")
        else:
            print_with_prefix(("Correct: The 'chexpert' directory exists in "
                            f"{target_dir} with {train_count} train files and "
                            f"{test_count} test files."))
    
    print_with_prefix('All datasets are prepared correctly.')

if __name__ == '__main__':

    path_dict = {
        'Chest-X-Ray14': '/data/jingfengyao/datasets/NIH',                          # path to unzipped CXR14
        'CovidX-CXR-3': '/data/jingfengyao/datasets/covidx_version5',               # path to unzipped CovidX-CXR-3 (version5)
        'CovidX-CXR-4': '/data/jingfengyao/datasets/covidx-cxr-4',                  # path to unzipped CovidX-CXR-4 (version6)
        'CheXpert': '/data/jingfengyao/datasets/chexpert/CheXpert-v1.0-small'       # path to unzipped CheXpert
    }
    tgt_root = 'datasets/'                                                          # all the prepared datasets will be stored here

    prepare_cxr14(path_dict['Chest-X-Ray14'], os.path.join(tgt_root, 'cxr14'))
    prepare_covidx3(path_dict['CovidX-CXR-3'], os.path.join(tgt_root, 'covidx3'))
    prepare_covidx4(path_dict['CovidX-CXR-4'], os.path.join(tgt_root, 'covidx4'))
    prepare_chexpert(path_dict['CheXpert'], os.path.join(tgt_root, 'chexpert'))

    check_dataset(tgt_root)