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
from torch.nn.modules.upsampling import Upsample

from torchsummary import summary

from models.hourglass import hg
from models.dense_net import densenet121


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Base_CNN(nn.Module):
    def __init__(self, filters=[64, 128, 256]):
        super(Base_CNN, self).__init__()

        self.maxpool2d = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        self.conv1_a = conv3x3(3, filters[0])
        self.conv1_b = conv3x3(filters[0], filters[0])

        self.conv2_a = conv3x3(filters[0], filters[1])
        self.conv2_b = conv3x3(filters[1], filters[1])

        self.conv3_a = conv3x3(filters[1], filters[2])
        self.conv3_b = conv3x3(filters[2], filters[2])

# self.linear = nn.Linear(, 1000)

    def forward(self, x):

        out = self.conv1_a(x)
        out = self.relu(out)
        out = self.conv1_b(out)
        out = self.relu(out)
        out = self.maxpool2d(out)

        out = self.conv2_a(out) 
        out = self.relu(out)
        out = self.conv2_b(out)
        out = self.relu(out)  
        out = self.maxpool2d(out)

        out = self.conv3_a(out)
        out = self.relu(out)
        out = self.conv3_b(out)
        out = self.relu(out)
        out = self.maxpool2d(out)

        out = torch.flatten(out, 1)

        return out


class E_Net(nn.Module):
    """
    E Net

    author : Dusan, Nikola, Nikhil
    """

    def __init__(self, device='cuda'):

        super(E_Net, self).__init__()
        
        self.relu = nn.ReLU()

        # E-Net
        self.e_net_base_left = Base_CNN()
        self.e_net_left_fc_1000 = nn.Linear(256 * 7 * 11, 1000)
        self.e_net_left_fc_500 = nn.Linear(1000, 500)

        self.e_net_base_right = Base_CNN()
        self.e_net_right_fc_1000 = nn.Linear(256 * 7 * 11, 1000)
        self.e_net_right_fc_500 = nn.Linear(1000, 500)

        self.e_net_end_part = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        left_eye_features = self.e_net_base_left(x['left-eye'])
        right_eye_features = self.e_net_base_right(x['right-eye'])

        left_eye_features = self.e_net_left_fc_1000(left_eye_features)
        left_eye_features = self.relu(left_eye_features)
        left_eye_features = self.e_net_left_fc_500(left_eye_features)
        left_eye_features = self.relu(left_eye_features)

        right_eye_features = self.e_net_right_fc_1000(right_eye_features)
        right_eye_features = self.relu(right_eye_features)
        right_eye_features = self.e_net_left_fc_500(right_eye_features)

        eye_features_concat = torch.cat(
            [left_eye_features, right_eye_features], dim=1)

        result = self.e_net_end_part(eye_features_concat)

        p_l = result[:,0]
        p_r = result[:,1]
        return {'p_l': p_l, 'p_r': p_r}


class AR_Net_Deep_Pictorial_Gaze(nn.Module):
    """
    AR-Net + Deep Pictorial Gaze


    description : Model adjusted from Appearance-Based Gaze Estimation via 
    Evaluation-Guided Asymmetric Regression + Deep Pictorial Gaze

    author : Dusan, Nikola, Nikhil
    """

    def __init__(self, device='cuda', multiple_intermediate_losses = False):

        super(AR_Net_Deep_Pictorial_Gaze, self).__init__()

        self.relu = nn.ReLU()

        self.multiple_intermediate_losses = multiple_intermediate_losses

        self.eye_upsample = Upsample(scale_factor=2.5, mode='bicubic')

        # AR-Net

        input_dict = {"num_stacks": 1,
                    "num_blocks": 1,
                    "num_classes": 2}  # TODO Check out how does this exactly work

        self.input_dict = input_dict
        self.eye_upsample = Upsample(scale_factor=2.5, mode='bicubic')

        self.stacked_hourglass = hg(**input_dict).to(device)

        self.dense_net = densenet121(pretrained=False, progress=False,input_channels_size = 2)



    def forward(self, x):
        left_eye = self.eye_upsample(x['left-eye'])
        right_eye = self.eye_upsample(x['right-eye'])
        # [n_batch, 3, 150, 225] --->  [n_batch, n_class, 37, 56]
        gaze_map_left = self.stacked_hourglass(left_eye)
        gaze_map_right = self.stacked_hourglass(right_eye)

        gaze_left_pred = self.dense_net(gaze_map_left[0])
        gaze_right_pred = self.dense_net(gaze_map_right[0])

        return {'gaze_left_pred': gaze_left_pred,
                'gaze_right_pred': gaze_right_pred}



class MultiModalUnit(nn.Module):
    """
        # TODO comment
        multiple_intermediate_losses (boolean) : checking if we need multiple intermediate losses in stacked hourglass model
    """
    def __init__(self, device = 'cuda', multiple_intermediate_losses = False):
        super(MultiModalUnit, self).__init__()  

        self.e_net = E_Net(device=device)
        self.ar_net = AR_Net_Deep_Pictorial_Gaze(device=device, multiple_intermediate_losses = multiple_intermediate_losses)

    def forward(self, x):
        e_net_result = self.e_net.forward(x)
        ar_net_result = self.ar_net.forward(x)

        return {**e_net_result, **ar_net_result}

