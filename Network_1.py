# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 15:02:45 2025

@author: scarl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from swin_unetr import SwinUNETR

class FC_SWIN(nn.Module):
    def __init__(self, input_dim=256*256, hidden_dim=16*256, output_dim=256*256, img_size=(256, 256), activation="tanh"):
        super(FC_SWIN, self).__init__()
        self.fc_input_dim = input_dim
        self.fc_hidden_dim = hidden_dim
        self.fc_output_dim = output_dim
        self.im_h, self.im_w = img_size

        self.fc_1 = nn.Linear(self.fc_input_dim, self.fc_hidden_dim)
        self.fc_2 = nn.Linear(self.fc_hidden_dim, self.fc_output_dim)

        # Define activation function
        self.activation = self.get_activation_function(activation)

        self.SwinUNETR = SwinUNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=1,
            feature_size=48,
            spatial_dims=2
        )

    def get_activation_function(self, activation):
        """Returns the selected activation function."""
        if activation.lower() == "relu":
            return F.relu
        elif activation.lower() == "leaky_relu":
            return F.leaky_relu
        elif activation.lower() == "tanh":
            return torch.tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation}. Choose from 'relu', 'leaky_relu', or 'tanh'.")

    def forward(self, x):
        x = x.view(x.size(0), -1)
        act1 = self.activation(self.fc_1(x))
        act2 = self.activation(self.fc_2(act1))
        #x = self.activation(self.fc_1(x))
        #x = self.activation(self.fc_2(x))
        
        fc = act2.view(-1, 1, self.im_h, self.im_w)  # Shape: (batch_size, 1, 256, 256)

        y = self.SwinUNETR(fc)  # Shape: (batch_size, 1, 256, 256)
        
        return y, fc, act1, act2
