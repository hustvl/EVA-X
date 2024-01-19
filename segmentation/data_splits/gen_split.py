import os

def save_image_ids(folder1, folder2, save_dir):
    train_file = os.path.join(save_dir, "train.txt")
    test_file = os.path.join(save_dir, "test.txt")

    # Create save_dir if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get all image IDs from folder1
    folder1_ids = [filename[:-4] for filename in os.listdir(folder1) if filename.endswith(".jpg") or filename.endswith(".png")]

    # Get all image IDs from folder2
    folder2_ids = [filename[:-4] for filename in os.listdir(folder2) if filename.endswith(".jpg") or filename.endswith(".png")]

    # Write image IDs to train.txt file
    with open(train_file, "w") as f:
        for image_id in folder1_ids:
            f.write(image_id + "\n")

    # Write image IDs to test.txt file
    with open(test_file, "w") as f:
        for image_id in folder2_ids:
            f.write(image_id + "\n")

# Call the function and pass the paths of two folders
save_image_ids("/home/jingfengyao/code/medical/pre-release/segmentation/dataset/Shenzhen/test", 
               "/home/jingfengyao/code/medical/pre-release/segmentation/dataset/Shenzhen/train",
               'data_splits/shenzhen')
