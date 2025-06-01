# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 23:43:03 2025

@author: scarl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AUTOMAP_BASIC_2(nn.Module):
    def __init__(self, config):
        super(AUTOMAP_BASIC_2, self).__init__()
        self.fc_input_dim = config.fc_input_dim
        self.fc_hidden_dim = config.fc_hidden_dim
        self.fc_output_dim = config.fc_output_dim
        self.im_h = config.im_h
        self.im_w = config.im_w
        
        self.fc_1 = nn.Linear(self.fc_input_dim, self.fc_hidden_dim)
        self.fc_2 = nn.Linear(self.fc_hidden_dim, self.fc_output_dim)
        
        self.conv_layers = nn.Sequential(
            #nn.ZeroPad2d(4),
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=7, stride=1, padding=3)
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
        # Device placement in PyTorch is handled outside of the model
        x = x.view(x.size(0), -1)
        act1 = self.activation(self.fc_1(x))
        act2 = self.activation(self.fc_2(act1))
        
        fc = act2.view(-1, 1, self.im_h, self.im_w)
        
        c_2 = self.conv_layers[1](self.conv_layers[0](fc))
        c_2 = self.conv_layers[3](self.conv_layers[2](c_2))
        #c_3 = self.conv_layers[5](self.conv_layers[4](c_2))
        c_3 = self.conv_layers[4](c_2)
        
        output = c_3.view(c_3.size(0), -1)
        
        return c_3, fc, output, act1#output#c_2, output