import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py
import matplotlib
import torch.nn as nn
import torch.optim as optim
from  torch.nn.modules.upsampling import Upsample

from torchsummary import summary

from models.hourglass import hg
from models.u_net import UNet
from models.dense_net import small_densenet, smaller_densenet

import torchvision.transforms as T

import torch.nn.functional as F


class MultiModalUnit(nn.Module):
    """
        # TODO comment
        multiple_intermediate_losses (boolean) : checking if we need multiple intermediate losses in stacked hourglass model
    """
    def __init__(self, sample_dictionary, net_1, num_stacks = 1,  device = 'cuda', multiple_intermediate_losses = False):
        super(MultiModalUnit, self).__init__()  
        UPSCALE = 3
        self.eye_upsample = Upsample(scale_factor=UPSCALE, mode='bicubic')
        self.net_1_name = net_1 

        if self.net_1_name == "H":
            input_dict = {"num_stacks": num_stacks,
                          "num_blocks": 1, 
                          "num_classes": 2}  # TODO Check out how does this exactly work

            self.net_1 = hg(**input_dict).to(device)
            self.net_2 = smaller_densenet()
        elif self.net_1_name == "U":
            assert UPSCALE == 3 # OTherwise, the resize dimensions are not appropriate....the final size needs to be divisable by 16
            self.net_1 = UNet(in_channels=3, out_channels=2, init_features=32).to(device)
            self.net_2 = small_densenet()
        else:
            raise AssertionError

        self.fc = nn.Linear(self.net_2.num_feature_channels(), 2)

        self.device = device
        self.multiple_intermediate_losses = multiple_intermediate_losses

        print(f'Is it multiple intermediate losses model: {multiple_intermediate_losses}')
        return 
        
    def forward(self, x):
        left_eye = self.eye_upsample(x['left-eye'])
        right_eye = self.eye_upsample(x['right-eye'])

        if self.net_1_name == "U":
            left_eye = F.pad(input=left_eye, pad=(1, 1, 6, 6), mode='constant', value=0.0) # 1,1 for last dim.....6,6 for second to last
            right_eye = F.pad(input=right_eye, pad=(1, 1, 6, 6), mode='constant', value=0.0) # 1,1 for last dim.....6,6 for second to last

        gaze_map_left = self.net_1(left_eye)  # [n_batch, 3, 150, 225] --->  [n_batch, n_class, 37, 56]
        gaze_map_right = self.net_1(right_eye)

        if self.net_1_name == "U":
            # Return to original dimensions and pack in list    
            gaze_map_left = [gaze_map_left[:, :, 6:-6, 1:-1]]
            gaze_map_right = [gaze_map_right[:, :, 6:-6, 1:-1]]

        gaze_map_concat = torch.cat([gaze_map_left[-1], gaze_map_right[-1]], dim=1)
        gaze_direction = self.net_2(gaze_map_concat) 
        gaze_direction = self.fc(gaze_direction)

        # gaze_map_left = self.net_1(left_eye)[-1]  # [n_batch, 3, 150, 225] --->  [n_batch, n_class, 37, 56]
        # gaze_map_right = self.net_1(right_eye)[-1]
        # gaze_map_concat = torch.cat([gaze_map_left, gaze_map_right], dim=1)
        # gaze_direction = self.net_2(gaze_map_concat)
        
        return gaze_map_left, gaze_map_right,  gaze_direction