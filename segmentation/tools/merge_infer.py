import os
import cv2

def merge_images(folder1, folder2, output_folder):
    # Get all image filenames in folder1
    filenames1 = os.listdir(folder1)

    # makedir if not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate through each image file in folder1
    for filename in filenames1:
        # Build file paths
        img_path1 = os.path.join(folder1, filename)
        img_path2 = os.path.join(folder2, filename)
        
        # Check if file paths exist
        if os.path.isfile(img_path1) and os.path.isfile(img_path2):
            # Read images
            img1 = cv2.imread(img_path1)
            img2 = cv2.imread(img_path2)
            
            # Check if image sizes are the same
            if img1.shape != img2.shape:
                # Resize images to the same size
                min_height = min(img1.shape[0], img2.shape[0])
                min_width = min(img1.shape[1], img2.shape[1])
                img1 = cv2.resize(img1, (min_width, min_height))
                img2 = cv2.resize(img2, (min_width, min_height))
            
            # Concatenate images
            merged_img = cv2.hconcat([img1, img2])
            
            # Build output file path
            output_path = os.path.join(output_folder, filename)
            
            # Save merged image
            cv2.imwrite(output_path, merged_img)
            
            print(f"Successfully merged {filename}")
        else:
            print(f"Skipping {filename} due to missing file")

if __name__ == "__main__":
    # Get folder paths
    folder1 = '/home/jingfengyao/code/medical/mmsegmentation/projects/unet_cxr/work_dirs/upernet_eva_x_adapter_small_siim_20k_eva_x_pt/inference'
    folder2 = '/home/jingfengyao/code/medical/mmsegmentation/projects/unet_cxr/work_dirs/upernet_vit_adapter_small_siim_20k_medicalmae/inference'
    output_folder = '../inference'
    
    # Merge images
    merge_images(folder1, folder2, output_folder)