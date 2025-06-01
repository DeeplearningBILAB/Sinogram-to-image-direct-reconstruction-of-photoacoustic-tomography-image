# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 19:59:30 2025

@author: scarl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from math import exp


def HybridLoss(pred, target, alpha = 0.5):
    mse_loss = nn.MSELoss()
    mse = mse_loss(pred, target)
    
    data_range = (target.max().cpu().numpy() - target.min().cpu().numpy())
    ssim_index, _ = ssim(pred, target, full=True, data_range=data_range)
    return alpha * ssim_index + (1 - alpha) * mse

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True):
    if window is None:
        channel = img1.size(1)
        window = create_window(window_size, channel)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class Combined_loss(torch.nn.Module):
    def __init__(self, alpha=0.5, window_size=11, size_average=True):
        super(Combined_loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.alpha = alpha

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
        
        self.window = window
        self.channel = channel
        
        ssimLoss = 1 - ssim(img1, img2, window=window, size_average=self.size_average)
        mse_loss = F.mse_loss(img1, img2)
        return self.alpha * ssimLoss + (1 - self.alpha) * mse_loss


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
        
        self.window = window
        self.channel = channel

        return 1 - ssim(img1, img2, window=window, size_average=self.size_average)
    
class Combined_SSIM_MAE(torch.nn.Module):
    def __init__(self, alpha=0.5, window_size=11, size_average=True):
        super(Combined_SSIM_MAE, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.alpha = alpha

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
        
        self.window = window
        self.channel = channel
        
        ssimLoss = 1 - ssim(img1, img2, window=window, size_average=self.size_average)
        #mse_loss = F.mse_loss(img1, img2)
        loss_function = nn.L1Loss()
        MAE_LOSS = loss_function(img1, img2)
        return self.alpha * ssimLoss + (1 - self.alpha) * MAE_LOSS

def Mae(predicted_x, labels_v):
    loss_function = nn.L1Loss()
    loss = loss_function(predicted_x, labels_v)
    return loss