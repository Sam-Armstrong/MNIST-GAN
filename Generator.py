"""
Author: Sam Armstrong
Date: Autumn 2021

Description: The generator model for the GAN
"""

import torch.nn as nn

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv2 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv3 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))
        self.conv4 = nn.Conv2d(in_channels = 6, out_channels = 9, kernel_size = (5, 5), stride = (1, 1), padding = (2, 2))
        # Pointwise convolution
        self.conv5 = nn.Conv2d(in_channels = 9, out_channels = 1, kernel_size = (1, 1), stride = (1, 1), padding = (0, 0))
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(784, 784)
        self.fc2 = nn.Linear(784, 784)
        self.fc3 = nn.Linear(784, 784)
        self.bn = nn.BatchNorm1d(784)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.fc3(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = x.reshape((x.shape[0], 1, 28, 28))
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv3(x)
        x = self.gelu(x)
        x = self.conv4(x)
        x = self.gelu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = x.reshape((x.shape[0], 1, 28, 28))
        return x
