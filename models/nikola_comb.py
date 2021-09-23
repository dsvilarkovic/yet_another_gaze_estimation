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

    def __init__(self, backbone_arch="resnet18", device='cuda'):
        super(MultiModalUnit, self).__init__()
        self.gaze_pred_net = Gaze_pred_Net(backbone_arch=backbone_arch).to(device)
        self.e_net = E_Net().to(device)
        return

    def forward(self, x):
        gaze = self.gaze_pred_net(left_eye=x['left-eye'],
                                  right_eye=x['right-eye'],
                                  head_pose_mask=x['face'],
                                  face_landmarks=x['face-landmarks'],
                                  head_pose_vec=x['head'])

        prob = self.e_net(left_eye_image=x['left-eye'], right_eye_image=x['right-eye'])
		
        gaze["final-pred"] = (gaze["gaze-left"] * prob['p-l']) + (gaze["gaze-right"] * prob['p-r']) # TODO Maybe change in the future
        # gaze["final-pred-equal"] = (gaze["gaze-left"]  + gaze["gaze-right"]) / 2.0

        return {**gaze, **prob}


class Gaze_pred_Net(nn.Module):

    def __init__(self, backbone_arch):
        super(Gaze_pred_Net, self).__init__()

        # Stteam 1 - eye image
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

        self.stream_1_fc = nn.Sequential(
            nn.Linear(self.eye_net.num_feature_channels(), 1000) , 
            nn.ReLU(inplace=True) ,
            nn.Dropout(0.5),
            
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # Stream 2 - head pose mask
        # TODO Currently Base Net because this feels like it needs a lightweight net....maybe MobileNet2???
        self.head_pose_net = Base_CNN()
        self.stream_2_fc = nn.Sequential(
            nn.Linear(self.head_pose_net.num_feature_channels(), 300) , 
            nn.ReLU(inplace=True) ,
            nn.Dropout(0.5),
            
            nn.Linear(300, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # Stream 3 - facial landmarks
        self.stream_3_fc = nn.Sequential(
            nn.Linear(66, 500) , 
            nn.ReLU(inplace=True) ,
            nn.Dropout(0.5),
            
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(500, 200), 
            nn.ReLU(inplace=True) ,
            nn.Dropout(0.5),
        )
        
        # Combine all streames together
        self.all_features_fc = nn.Sequential(
            nn.Linear(902, 900) , 
            nn.ReLU(inplace=True) ,
            nn.Dropout(0.5),
            
            nn.Linear(900, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # Predict gaze from the final feature map
        self.pred_gaze = nn.Linear(500, 2)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout5 = nn.Dropout(p=0.5)


    def forward(self, left_eye, right_eye, head_pose_mask, face_landmarks, head_pose_vec):

        s1_out_left = self.eye_net(left_eye)
        s1_out_left = self.stream_1_fc(s1_out_left)

        s1_out_right = self.eye_net(right_eye)
        s1_out_right = self.stream_1_fc(s1_out_right)

        head_pose_mask_norm = 2.0 * (head_pose_mask - 0.5) # Normalize input data
        s2_out = self.head_pose_net(head_pose_mask_norm)
        s2_out = self.stream_2_fc(s2_out)

        batch_size, _, _ = face_landmarks.size()
        face_landmarks_norm = face_landmarks.float() / 112.0 - 1.0 # Normalize input data
        s3_out = self.stream_3_fc(face_landmarks_norm.view(batch_size, -1))

        left_gaze_interm = torch.cat((s1_out_left, s2_out, s3_out, head_pose_vec.float()), dim=1)
        left_gaze_interm = self.all_features_fc(left_gaze_interm)
        left_gaze_pred = self.pred_gaze(left_gaze_interm)

        right_gaze_interm = torch.cat((s1_out_right, s2_out, s3_out, head_pose_vec.float()), dim=1)
        right_gaze_interm = self.all_features_fc(right_gaze_interm)
        right_gaze_pred = self.pred_gaze(right_gaze_interm)

        gaze = {'gaze-left':  left_gaze_pred,
                'gaze-right': right_gaze_pred,
                }

        return gaze

class E_Net(nn.Module):

    def __init__(self):
        super(E_Net, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.dropout5 = nn.Dropout(p=0.5)

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

        self.num_feature_ch = filters[-1]

        self.maxpool2d = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        self.conv1_a = conv3x3(3, filters[0])
        self.conv1_b = conv3x3(filters[0], filters[0])

        self.conv2_a = conv3x3(filters[0], filters[1])
        self.conv2_b = conv3x3(filters[1], filters[1])

        self.conv3_a = conv3x3(filters[1], filters[2])
        self.conv3_b = conv3x3(filters[2], filters[2])

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # self.linear = nn.Linear(, 1000)

    def num_feature_channels(self):
        return self.num_feature_ch

    def forward(self, x):

        x = self.conv1_a(x)
        x = self.relu(x)
        x = self.conv1_b(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        x = self.conv2_a(x)
        x = self.relu(x)
        x = self.conv2_b(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        x = self.conv3_a(x)
        x = self.relu(x)
        x = self.conv3_b(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        return x
