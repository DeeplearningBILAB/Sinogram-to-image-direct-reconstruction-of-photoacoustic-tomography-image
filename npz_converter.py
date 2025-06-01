# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 15:45:54 2025

@author: scarl
"""

import os
import numpy as np
import re
from scipy.io import loadmat, savemat
from utils import image_batch_norm
from skimage.transform import resize
def process_dataset_train(input_directory, GT_directory, save_dir, input_variable="PAT_data", GT_variable="normalized_img", batch_norm = None, number_img=8000, input_img_shape = None, gt_img_shape = None):
    """
    Processes the dataset by loading .mat files, normalizing images, and saving them as .npz files.

    Args:
        input_directory (str): Path to the input sinogram .mat files.
        GT_directory (str): Path to the ground truth .mat files.
        save_dir (str): Path to save the processed .npz files.
        input_variable (str): Variable name for the input sinogram images. Default is "PAT_data".
        GT_variable (str): Variable name for the ground truth images. Default is "normalized_img".
        number_img (int): Number of images to process. Default is 8000.
        img_shape = (256, 256, 1)
    """

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get all .mat files and sort numerically
    mat_files = sorted(
        [f for f in os.listdir(input_directory) if f.endswith('.mat')],
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf')
    )
    mat_files = mat_files[:number_img]

    # Split dataset into train/val/test
    total_index = np.random.permutation(len(mat_files)) # return shuffled index
    train_size, val_size = int(0.7 * len(total_index)), int(0.15 * len(total_index))
    train_idx, val_idx, test_idx = np.split(total_index, [train_size, train_size + val_size])

    subsets = {
        "Training": train_idx,
        "Validation": val_idx,
        "Test": test_idx
    }


    for subset, indices in subsets.items():
        input_list, gt_list = [], []

        for i, idx in enumerate(indices):
            print(i)
            mat_name = mat_files[idx]

            # Load input data
            input_data = loadmat(os.path.join(input_directory, mat_name))[input_variable]
            input_data = input_data.astype(np.float32)
            #input_data = input_data[0:512:2, 0:1024:4].astype(np.float32)  # Downsample
            #input_data = np.reshape(input_data, img_shape)

            # Load ground truth
            gt_data = loadmat(os.path.join(GT_directory, mat_name))[GT_variable]
            gt_data = gt_data.astype(np.float32)
            if input_img_shape is not None:
                input_data = resize(input_data, input_img_shape, mode='reflect', anti_aliasing=True)
            if gt_img_shape is not None:    
                gt_data = resize(gt_data, gt_img_shape, mode='reflect', anti_aliasing=True)

            # Store in lists
            input_list.append(input_data)
            gt_list.append(gt_data)

            # Print progress every 500 steps
            if (i + 1) % 500 == 0:
                print(f"{subset} Processing: {i+1}/{len(indices)} images")

        if batch_norm == True:
            input_training = np.asarray(input_list)
            gt_training = np.asarray(gt_list)

            input_training = image_batch_norm(input_training)
            gt_training = image_batch_norm(gt_training)
        else:
            input_training = input_list
            gt_training = gt_list

        Training_data = list(zip(input_training, gt_training))
        input_training, gt_training = zip(*Training_data)
        np.savez(os.path.join(save_dir, f"{subset}.npz"), np.asarray(input_training), np.asarray(gt_training))

    # Save test set indices to .mat
    test_filenames = [mat_files[i] for i in test_idx]
    test_numbers = [int(re.search(r'\d+', f).group()) for f in test_filenames if re.search(r'\d+', f)]
    savemat(os.path.join(save_dir, 'Test_index.mat'), {'test_array': test_numbers})

    print("All datasets processed and saved successfully!")
    
def process_dataset_test(input_directory, GT_directory, save_dir, input_variable="PAT_data", GT_variable="normalized_img", number_img=8000, input_img_shape = None, gt_img_shape = None):
    """
    Processes the dataset by loading .mat files, normalizing images, and saving them as .npz files.

    Args:
        input_directory (str): Path to the input sinogram .mat files.
        GT_directory (str): Path to the ground truth .mat files.
        save_dir (str): Path to save the processed .npz files.
        input_variable (str): Variable name for the input sinogram images. Default is "PAT_data".
        GT_variable (str): Variable name for the ground truth images. Default is "normalized_img".
        number_img (int): Number of images to process. Default is 8000.
        img_shape = (256, 256)
    """

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get all .mat files and sort numerically
    mat_files = sorted(
        [f for f in os.listdir(input_directory) if f.endswith('.mat')],
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf')
    )
    mat_files = mat_files[:number_img]
    print(mat_files)

    input_list, gt_list = [], []
    indices = np.arange(0,number_img)

    for idx in range(number_img):
        print(idx)
        mat_name = mat_files[idx]

        # Load input data
        input_data = loadmat(os.path.join(input_directory, mat_name))[input_variable]
        input_data = input_data.astype(np.float32)
        #input_data = input_data[0:512:2, 0:1024:4].astype(np.float32)  # Downsample
        #input_data = np.reshape(input_data, img_shape)

        # Load ground truth
        gt_data = loadmat(os.path.join(GT_directory, mat_name))[GT_variable]
        gt_data = gt_data.astype(np.float32)
        if input_img_shape is not None:
            input_data = resize(input_data, input_img_shape, mode='reflect', anti_aliasing=True)
        if gt_img_shape is not None:    
            gt_data = resize(gt_data, gt_img_shape, mode='reflect', anti_aliasing=True)

        # Store in lists
        input_list.append(input_data)
        gt_list.append(gt_data)
    Training_data = list(zip(input_list, gt_list))
    input_training, gt_training = zip(*Training_data)
    np.savez(os.path.join(save_dir, f"Exptest.npz"), np.asarray(input_training), np.asarray(gt_training))  
  

# Example usage
if __name__ == "__main__":
    npz_dir = "G:\\npz_data"
    data_dir = "G:\\mat"
    input_dir = os.path.join(data_dir, "sinogram")
    gt_dir = os.path.join(data_dir, "recon")
    process_dataset_test(
        input_directory=input_dir,
        GT_directory=gt_dir,
        save_dir=npz_dir,
        input_variable="PAT_data",
        GT_variable="normalized_img",
        batch_norm = True,
        number_img=8000
    )
