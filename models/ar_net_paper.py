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

from models.mobilenet2_head import mobilenet_v2
from models.resnet_head import resnet18, resnet34, resnet50
from models.dense_net_head import small_densenet, medium_densenet, densenet121, densenet161

from torchsummary import summary


class MultiModalUnit(nn.Module):
    """
    """

    def __init__(self, backbone_arch="mnet2", device='cuda'):
        super(MultiModalUnit, self).__init__()
        self.ar_net = AR_Net(backbone_arch=backbone_arch).to(device)
        self.e_net = E_Net(backbone_arch=backbone_arch).to(device)
        return

    def forward(self, x):
        # TODO Is 'head' the thing we were looking for?
        gaze = self.ar_net(left_eye_image=x['left-eye'], right_eye_image=x['right-eye'], head_pose=x['head'])

        prob = self.e_net(left_eye_image=x['left-eye'], right_eye_image=x['right-eye'])
        p_r = prob['p-r']
        p_l = prob['p-l']
		
        gaze["final-pred"] = (gaze["gaze-left"] * p_l) + (gaze["gaze-right"] * p_r) # TODO Maybe change in the future
        # gaze["final-pred-equal"] = (gaze["gaze-left"]  + gaze["gaze-right"]) / 2.0

        return {**gaze, **prob}


class AR_Net(nn.Module):
    """
    """

    def __init__(self, backbone_arch):
        super(AR_Net, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.dropout5 = nn.Dropout(p=0.5)

        if backbone_arch == "mnet2":
            self.eye_net = mobilenet_v2(pretrained=True)
        elif backbone_arch == "base":
            self.eye_net = Base_CNN()
        elif backbone_arch == "resnet18":
            self.eye_net = resnet18(pretrained=True, progress=False)
        elif backbone_arch == "resnet34":
            self.eye_net = resnet34(pretrained=True, progress=False)
        elif backbone_arch == "resnet50":
            self.eye_net = resnet50(pretrained=True, progress=False)
        elif backbone_arch == "sdensenet":
            self.eye_net = small_densenet() # Doesnt have the option to load pretrained weights
        elif backbone_arch == "mdensenet":
             self.eye_net = medium_densenet() # Doesnt have the option to load pretrained weights
        elif backbone_arch == "densenet121":
            self.eye_net = densenet121(pretrained=True, progress=False)
        elif backbone_arch == "densenet161":
            self.eye_net = densenet161(pretrained=True, progress=False)
        else:
            raise NotImplementedError

        self.stream_1_fc_1 = nn.Linear(self.eye_net.num_feature_channels(), 1000)
        self.stream_1_fc_2 = nn.Linear(1000, 500)

        self.stream_2_fc_1 = nn.Linear(self.eye_net.num_feature_channels(), 1000)
        self.stream_2_fc_2 = nn.Linear(1000, 500)

        self.stream_3_fc_1 = nn.Linear(self.eye_net.num_feature_channels(), 500)

        self.stream_4_fc_1 = nn.Linear(self.eye_net.num_feature_channels(), 500)

        self.fc_3_4 = nn.Linear(1000, 500)

        self.fc_last = nn.Linear(1502, 4)

    def forward(self, left_eye_image, right_eye_image, head_pose):

        s1_out = self.eye_net(left_eye_image)
        s1_out = self.stream_1_fc_1(s1_out)
        s1_out = self.relu(s1_out)
        s1_out = self.dropout5(s1_out)
        s1_out = self.stream_1_fc_2(s1_out)
        s1_out = self.relu(s1_out)
        s1_out = self.dropout5(s1_out)

        s2_out = self.eye_net(right_eye_image)
        s2_out = self.stream_2_fc_1(s2_out)
        s2_out = self.relu(s2_out)
        s2_out = self.dropout5(s2_out)
        s2_out = self.stream_2_fc_2(s2_out)
        s2_out = self.relu(s2_out)
        s2_out = self.dropout5(s2_out)

        s3_out = self.eye_net(left_eye_image)
        s3_out = self.stream_3_fc_1(s3_out)
        s3_out = self.relu(s3_out)
        s3_out = self.dropout5(s3_out)

        s4_out = self.eye_net(right_eye_image)
        s4_out = self.stream_4_fc_1(s4_out)
        s4_out = self.relu(s4_out)
        s4_out = self.dropout5(s4_out)

        s_3_4_out = torch.cat((s3_out, s4_out), dim=1)
        s_3_4_out = self.fc_3_4(s_3_4_out)
        s_3_4_out = self.relu(s_3_4_out)
        s_3_4_out = self.dropout5(s_3_4_out)
        

        out = torch.cat((s1_out, s2_out, s_3_4_out,
                        head_pose.type(torch.float32)), dim=1)
        out = self.fc_last(out)
        # TODO Maybe one more layer here to further process the raw head pose

        gaze = {'gaze-left':  out[:, :2],
                'gaze-right': out[:, 2:],
                }

        return gaze


class E_Net(nn.Module):

    def __init__(self, backbone_arch):
        super(E_Net, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.dropout5 = nn.Dropout(p=0.5)

        # if backbone_arch == "mnet2":
        #     self.eye_net = mobilenet_v2(pretrained=True)
        # elif backbone_arch == "base":
        #     self.eye_net = Base_CNN()
        # else:
        #     raise NotImplementedError

        self.eye_net = mobilenet_v2(pretrained=True)

        self.sigmoid = nn.Sigmoid()

        self.stream_1_fc_1 = nn.Linear(self.eye_net.num_feature_channels(), 1000)
        self.stream_1_fc_2 = nn.Linear(1000, 500)

        self.stream_2_fc_1 = nn.Linear(self.eye_net.num_feature_channels(), 1000)
        self.stream_2_fc_2 = nn.Linear(1000, 500)

        self.fc_1_2 = nn.Linear(1000, 1)

    def forward(self, left_eye_image, right_eye_image):

        s1_out = self.eye_net(left_eye_image)
        s1_out = self.stream_1_fc_1(s1_out)
        s1_out = self.relu(s1_out)
        s1_out = self.dropout5(s1_out)
        s1_out = self.stream_1_fc_2(s1_out)
        s1_out = self.relu(s1_out)
        s1_out = self.dropout5(s1_out)

        s2_out = self.eye_net(right_eye_image)
        s2_out = self.stream_2_fc_1(s2_out)
        s2_out = self.relu(s2_out)
        s2_out = self.dropout5(s2_out)
        s2_out = self.stream_2_fc_2(s2_out)
        s2_out = self.relu(s2_out)
        s2_out = self.dropout5(s2_out)

        out = torch.cat((s1_out, s2_out), dim=1)
        out = self.fc_1_2(out)
        out_sigmoid = self.sigmoid(out)

        prob = {'p-r': 1.0 - out_sigmoid,
                'p-l': out_sigmoid,
                'p-raw': out,
                }

        return prob


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
