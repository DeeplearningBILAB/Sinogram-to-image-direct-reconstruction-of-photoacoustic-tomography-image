# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 20:15:50 2025

@author: scarl
"""

import numpy as np
import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def create_dataloader(base_dir, dataset_files, data_type, num_batch, num_workers=0, pin_memory=True):
        dataset_path = os.path.join(base_dir, dataset_files[data_type])
        dataset = CustomDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=num_batch, shuffle=True if data_type == "train" else False, num_workers=num_workers, pin_memory=pin_memory)
        return dataloader

class CustomDataset(Dataset):
    def __init__(self, npz_file):
        """
        Args:
            npz_file (str): Path to the dataset file.
            noise_type (str): Type of noise to add ('gaussian' or 'poisson'). Default is None (no noise).
            noise_level (float): Noise intensity (std for Gaussian, scaling factor for Poisson).
        """
        data = np.load(npz_file)
        self.images = data['arr_0']
        self.labels = data['arr_1']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # Convert to tensor and permute dimensions
        image = torch.from_numpy(image).permute(2, 0, 1)#.float()
        label = torch.from_numpy(label).permute(2, 0, 1)#.float()

        return {'image': image, 'label': label}


class CustomDataset_noise(Dataset):
    def __init__(self, npz_file, noise_type=None, noise_level=0.05, mask_count = None, pepper_prob=0.0):
        """
        Args:
            npz_file (str): Path to the dataset file.
            noise_type (str): Type of noise to add ('gaussian' or 'poisson'). Default is None (no noise).
            noise_level (float): Noise intensity (std for Gaussian, scaling factor for Poisson).
        """
        data = np.load(npz_file)
        self.images = data['arr_0']
        self.labels = data['arr_1']
        self.noise_type = noise_type
        self.noise_level = noise_level  # Standard deviation for Gaussian, scaling factor for Poisson
        self.mask_count = mask_count # Small mask size
        self.pepper_prob = pepper_prob  # Probability of adding pepper noise

    def add_gaussian_noise(self, image):
        """ Adds Gaussian noise with zero mean and specified standard deviation """
        noise = self.noise_level * torch.randn_like(image)
        return torch.clamp(image + noise, 0, 1)  # Clamp to [0,1] to prevent artifacts

    def add_poisson_noise(self, image):
        """ Adds Poisson noise by scaling image, applying Poisson, and normalizing """
        scale = 255.0  # Scale image to [0, 255] before applying Poisson noise
        noisy_image = torch.poisson(image * scale) / scale  # Normalize back to [0,1]
        return torch.clamp(noisy_image, 0, 1)
    
    def add_pepper_noise(self, image):
        """ Adds pepper noise by setting random pixels to zero with probability `pepper_prob` """
        if self.pepper_prob > 0:
            mask = torch.rand_like(image) < self.pepper_prob
            image[mask] = 0
        return image

    def __len__(self):
        return len(self.images)
    
    def _apply_mask(self, image):
        """ Applies a small random square mask to the image. """
        _, H, W, C = image.shape  # Get dimensions (C, H, W)

        # Ensure mask is smaller than the image
        if self.mask_count is None or self.mask_count <= 0:
            return image  # Skip masking if invalid

        # Randomly select top-left corner of the mask
        y_coords = np.random.randint(0, H, self.mask_count)
        x_coords = np.random.randint(0, W, self.mask_count)
        c_coords = np.random.randint(0, C, self.mask_count)

        # Apply the mask (set selected pixels to zero)
        image[c_coords, y_coords, x_coords] = 0

        return image


    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # Convert to tensor and permute dimensions
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        label = torch.from_numpy(label).permute(2, 0, 1).float()

        # Apply noise if specified
        if self.noise_type == 'gaussian':
            image = self.add_gaussian_noise(image)
        elif self.noise_type == 'poisson':
            image = self.add_poisson_noise(image)
        elif self.noise_type == 'pepper':
            image = self.add_pepper_noise(image)
        
        if self.mask_size is not None:
            image = self.apply_mask(image)

        return {'image': image, 'label': label}
