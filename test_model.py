# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 15:26:39 2025

@author: scarl
"""

import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
from scipy.io import savemat
from CustomData import CustomDataset_noise
from Network_1 import FC_SWIN
from Loss_function import Mae#, ssim
from utils import tensor_display, psnr, img_display

def test_model(
    test_loader, model_test, model_path, 
    im_save_dir_prediction, im_save_dir_GT, im_save_fc,
    batch_size=1, activation='relu', 
    device_idx=0, activation_display = None, 
    im_save_act1 = None, im_save_act2 = None,saving_enable=1
):
    """ 
    Tests a trained model on a given test dataset.
    
    Args:
        test_filename (str): Path to the test dataset (.npz).
        model_path (str): Path to the trained model (.pth).
        im_save_dir_prediction (str): Path to save predicted images.
        im_save_dir_GT (str): Path to save ground truth images.
        im_save_fc (str): Path to save FC layer outputs.
        batch_size (int, optional): Batch size for testing. Default is 1.
        activation (str, optional): Activation function type ('relu', 'tanh', 'leaky_relu'). Default is 'relu'.
        device_idx (int, optional): GPU device index. Default is 0.
    """
    # path creation
    if activation_display is None:
        dirs_to_create = [im_save_dir_prediction, im_save_dir_GT, im_save_fc]
    else:
        dirs_to_create = [im_save_dir_prediction, im_save_dir_GT, im_save_fc, im_save_act1, im_save_act2]
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)

    # Set device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # Enable cuDNN optimization

    # Define model
    #model_test.load_state_dict(torch.load(model_path))
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_test.load_state_dict(torch.load(model_path, map_location=map_location))
    model_test.eval()

    # Test loop
    total_loss = 0
    ssim_v = 0
    psnr_v = 0
    i = 1
    losses = []
    ssims = []
    psnrs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", dynamic_ncols=True):
            start_time = time.time()
            img = batch["image"].to(device, dtype=torch.float32)
            groundtruth = batch["label"].to(device, dtype=torch.float32)

            # Forward pass
            predicted_tx, predicted_tfc, preact1, preact2 = model_test(img)
            print(img.cpu().shape)

            # Compute loss (MAE)
            loss_v = Mae(predicted_tx, groundtruth)
            #total_loss += loss_v.item()
            losses.append(loss_v.item())
 
            # Compute SSIM and PSNR
            A, B = img_as_float(predicted_tx.cpu().squeeze()), img_as_float(groundtruth.cpu().squeeze())
            #ssim_v += ssim(A, B)
            ssim_index, _ = ssim(A, B, data_range=A.max() - A.min(), full=True)
            #ssim_v += ssim_index

            #psnr_v += psnr(A, B)
            ssims.append(ssim_index)
            psnrs.append(psnr(A, B))
            
            print(f"[Image {i:03d}] L1 Loss: {loss_v.item():.4f}, SSIM: {ssim_index:.4f}, PSNR: {psnr(A, B):.2f} dB, Time: {time.time() - start_time:.2f} sec")

            # Save results (only first 80 images)
            if saving_enable ==1:
                if i < 80:
                    #print(predicted_tx.cpu().shape)
                    tensor_display(predicted_tx.cpu(), 1, im_save_dir_prediction, name=i)
                    tensor_display(groundtruth.cpu(), 1, im_save_dir_GT, name=i)
                    tensor_display(predicted_tfc.cpu(), 1, im_save_fc, name=i)
                
                if activation_display is not None: 
                    preact1 = preact1.cpu().numpy().reshape(64, 64)
                    #print(preact1.size(0))
                    preact2 = preact2.cpu().numpy().reshape(256, 256)
                    #preact1 = preact1.cpu().numpy().reshape(64, 64)
                    #print(preact2.cpu().shape)
                    #preact2 = preact2.view(64, 64)
                    #img_display(preact1, 1, im_save_act1, name=i)
                    #img_display(preact2, 1, im_save_act2, name=i)
                savemat(os.path.join(im_save_dir_prediction, f'predicted_tx_{i}.mat'), {'predicted_tx': predicted_tx.cpu().numpy()})
                savemat(os.path.join(im_save_dir_GT, f'GT_{i}.mat'), {'GT': groundtruth.cpu().numpy()})

            # Save MATLAB files
            
            i += 1

    # Print summary metrics
    num_samples = i - 1
    losses = np.array(losses)
    ssims = np.array(ssims)
    psnrs = np.array(psnrs)

    # Print mean and standard deviation
    print(f"\nEvaluation Summary ({len(losses)} samples):")
    print(f"L1 Loss   - Mean: {losses.mean():.6f}, Std: {losses.std():.6f}")
    print(f"SSIM      - Mean: {ssims.mean():.6f}, Std: {ssims.std():.6f}")
    print(f"PSNR (dB) - Mean: {psnrs.mean():.6f}, Std: {psnrs.std():.6f}")
    # Find top 10 SSIM values and their indices
    top_10_indices = np.argsort(ssims)[-10:][::-1]
    top_10_ssims = ssims[top_10_indices]
    top_10_losses = losses[top_10_indices]
    top_10_psnrs = psnrs[top_10_indices]

    print("\nTop 6 SSIM Values and Corresponding Image Metrics:")
    print(" Rank | Image Index |     SSIM     |    L1 Loss   |   PSNR (dB)")
    print("---------------------------------------------------------------")
    for rank, (idx, ssim_val, l1_val, psnr_val) in enumerate(zip(top_10_indices, top_10_ssims, top_10_losses, top_10_psnrs), 1):
        print(f"  {rank:2d}   |    {idx+1:03d}     |  {ssim_val:.6f}  |  {l1_val:.6f}  |  {psnr_val:.2f}")
    print(f"\nselected output:")
    print(f"L1 Loss   - Mean: {top_10_losses.mean():.6f}, Std: {top_10_losses.std():.6f}")
    print(f"SSIM      - Mean: {top_10_ssims.mean():.6f}, Std: {top_10_ssims.std():.6f}")
    print(f"PSNR (dB) - Mean: {top_10_psnrs.mean():.6f}, Std: {top_10_psnrs.std():.6f}")