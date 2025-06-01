# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 22:46:02 2025

@author: scarl
"""
from npz_converter import process_dataset_test, process_dataset_train
import os


mode = 'train'
if mode == 'exp_test':
    base_path = 'G:\PhD\Third_year\Sinogram_recon\Experimental_data\All_text'
    save_dir = 'G:\PhD\Third_year\Sinogram_recon\\2Simulation_data\\20250319Data\\npz'
    process_dataset_test(base_path, base_path, 
                         save_dir, number_img=4, 
                         input_img_shape = (256,1024,1), 
                         gt_img_shape = (256, 256,1))
elif mode == 'train': 
    base_path= 'G:\PhD\Third_year\Sinogram_recon\\2Simulation_data\\20250513Data\\mat'
    save_dir = 'G:\PhD\Third_year\Sinogram_recon\\2Simulation_data\\20250513Data\\npz'
    input_directory = os.path.join(base_path, 'sinogram')
    GT_directory = os.path.join(base_path, 'recon')
    process_dataset_train(input_directory, GT_directory, 
                          save_dir, number_img=4500, 
                          input_img_shape = (256,1024,1), 
                          gt_img_shape = (256, 256,1))
elif mode == 'CT': 
    base_path= 'G:\PhD\Third_year\Sinogram_recon\\2Simulation_data\\20250331CT_data\\mat'
    save_dir = 'G:\PhD\Third_year\Sinogram_recon\\2Simulation_data\\20250331CT_data\\npz'
    input_directory = os.path.join(base_path, 'CT_sinogram')
    GT_directory = os.path.join(base_path, 'CT_source')
    process_dataset_train(input_directory, GT_directory, 
                          save_dir, number_img=5892, 
                          input_img_shape = (256,256,1), 
                          gt_img_shape = (256,256,1), 
                          input_variable="sinogram",
                          GT_variable="gt")
elif mode == 'UnseenTest':
    base_path = 'G:\PhD\Third_year\Sinogram_recon\\2Simulation_data\\Unseen_data\\New_v1\\mat'
    save_dir = 'G:\PhD\Third_year\Sinogram_recon\\2Simulation_data\\Unseen_data\\New_v1\\noisy_npz'
    input_directory = os.path.join(base_path, 'noisy_sinogram')
    GT_directory = os.path.join(base_path, 'recon')
    process_dataset_test(input_directory, GT_directory, 
                         save_dir, number_img=12, 
                         input_img_shape = (256,1024,1), 
                         gt_img_shape = (256, 256,1))
elif mode == 'UnseenTest_clean':
    base_path = 'G:\PhD\Third_year\Sinogram_recon\\2Simulation_data\\Unseen_data\\New_v1\\mat'
    save_dir = 'G:\PhD\Third_year\Sinogram_recon\\2Simulation_data\\Unseen_data\\New_v1\\npz_downsample'
    input_directory = os.path.join(base_path, 'sinogram_downsample')
    GT_directory = os.path.join(base_path, 'recon_downsample')
    process_dataset_test(input_directory, GT_directory, 
                         save_dir, number_img=52, 
                         input_img_shape = (256,1024,1), 
                         gt_img_shape = (256, 256,1))    