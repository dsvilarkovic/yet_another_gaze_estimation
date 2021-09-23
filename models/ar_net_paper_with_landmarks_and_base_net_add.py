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

    def __init__(self, backbone_arch="mnet2", device='cuda', pretrained=True):
        super(MultiModalUnit, self).__init__()
<<<<<<< HEAD
        self.ar_net = AR_Net(backbone_arch=backbone_arch, pretrained=pretrained, device = device).to(device)
        self.e_net = E_Net(backbone_arch=backbone_arch).to(device)
        self.device = device
=======
        self.ar_net = AR_Net(backbone_arch=backbone_arch, pretrained=pretrained).to(device)
        self.e_net = E_Net(backbone_arch=backbone_arch).to(device)
>>>>>>> 667e376d65e10a8618272c1b1f47f6cb81ed789d
        self.pretrained = pretrained
        return

    def forward(self, x):
        # TODO Is 'head' the thing we were looking for?
        gaze = self.ar_net(left_eye_image=x['left-eye'], right_eye_image=x['right-eye'], head_pose=x['head'], face_landmarks = {'face-landmarks' : x['face-landmarks']}, head_pose_mask=x['face'])

        prob = self.e_net(left_eye_image=x['left-eye'], right_eye_image=x['right-eye'])
        p_r = prob['p-r']
        p_l = prob['p-l']
		
        gaze["final-pred"] = (gaze["gaze-left"] * p_l) + (gaze["gaze-right"] * p_r) # TODO Maybe change in the future
        # gaze["final-pred-equal"] = (gaze["gaze-left"]  + gaze["gaze-right"]) / 2.0

        return {**gaze, **prob}


class AR_Net(nn.Module):
    """
    """

<<<<<<< HEAD
    def __init__(self, backbone_arch, pretrained = True, device = 'cuda'):
=======
    def __init__(self, backbone_arch, pretrained = True):
>>>>>>> 667e376d65e10a8618272c1b1f47f6cb81ed789d
        super(AR_Net, self).__init__()

        self.device = device

        self.relu = nn.ReLU(inplace=True)
        self.dropout5 = nn.Dropout(p=0.5)

        if backbone_arch == "mnet2":
            self.eye_net = mobilenet_v2(pretrained=pretrained)
        elif backbone_arch == "base":
            self.eye_net = Base_CNN()
        elif backbone_arch == "resnet18":
            self.eye_net = resnet18(pretrained=pretrained, progress=False)
        elif backbone_arch == "resnet34":
            self.eye_net = resnet34(pretrained=pretrained, progress=False)
        elif backbone_arch == "resnet50":
            self.eye_net = resnet50(pretrained=pretrained, progress=False)
        elif backbone_arch == "sdensenet":
            self.eye_net = small_densenet() # Doesnt have the option to load pretrained weights
        elif backbone_arch == "mdensenet":
             self.eye_net = medium_densenet() # Doesnt have the option to load pretrained weights
        elif backbone_arch == "densenet121":
            self.eye_net = densenet121(pretrained=pretrained, progress=False)
        elif backbone_arch == "densenet161":
            self.eye_net = densenet161(pretrained=pretrained, progress=False)
        else:
            raise NotImplementedError

        self.stream_1_fc_1 = nn.Linear(self.eye_net.num_feature_channels(), 1000)
        self.stream_1_fc_2 = nn.Linear(1000, 500)

        self.stream_2_fc_1 = nn.Linear(self.eye_net.num_feature_channels(), 1000)
        self.stream_2_fc_2 = nn.Linear(1000, 500)

        self.stream_3_fc_1 = nn.Linear(self.eye_net.num_feature_channels(), 500)

        self.stream_4_fc_1 = nn.Linear(self.eye_net.num_feature_channels(), 500)

        self.fc_3_4 = nn.Linear(1000, 500)


<<<<<<< HEAD
        self.landmark_unit = LandmarkUnit(feature_size=66, device=self.device)
=======
        self.landmark_unit = LandmarkUnit(feature_size=66, device='cuda' if torch.cuda.is_available() else "cpu")
>>>>>>> 667e376d65e10a8618272c1b1f47f6cb81ed789d
        self.base_net_additional = Base_CNN()

        # self.fc_last = nn.Linear(1502, 4)
        # self.fc_last = nn.Linear(1502 + 16 + 256, 4) # + 16 for landmakr_unit_output and 256 for Base_CNN output
        self.fc_last = nn.Linear(1502 + 128 + 256, 4) # + 16 for landmakr_unit_output and 256 for Base_CNN output

    def forward(self, left_eye_image, right_eye_image, head_pose, face_landmarks, head_pose_mask):

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
        

        ## TODO: New stuff, added for better result
        face_landmarks_out = self.landmark_unit(face_landmarks)
        head_pose_mask_out = self.base_net_additional(head_pose_mask)
        ###

        out = torch.cat((s1_out, s2_out, s_3_4_out,
                        head_pose.type(torch.float32), face_landmarks_out, head_pose_mask_out), dim=1)
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
        # print("NO ENET VERSION !!!!")

        # batch_size, _,_,_ = left_eye_image.shape

        # p = torch.tensor(batch_size * [0.5]).view(batch_size,-1).cuda()
        # prob = {'p-r': p,
        #         'p-l': p,
        #         'p-raw': p,
        #         }

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
        self.avgpool = nn.AdaptiveAvgPool2d(1)

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

        out = self.avgpool(out)

        out = torch.flatten(out, 1)


        return out


class LandmarkUnit(nn.Module):
    """
        Landmark units to be trained on, there are 33x2 landmarks
        
        inputs:
             feature_size (int) : size of features nx2
             output_size (int) : size to be output from this model
    """
    def __init__(self, feature_size = 66, output_size = 2, device = 'cuda'):
        super(LandmarkUnit, self).__init__()

        self.device = device
        self.feature_size = feature_size

        self.feature_model = nn.Sequential(
            nn.Linear(feature_size, int(256)),
            nn.BatchNorm1d(int(256)),
            nn.ReLU(inplace=True),
<<<<<<< HEAD
            nn.Dropout(0.5),

            nn.Linear(int(256), int(256)),
            nn.BatchNorm1d(int(256)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

=======
            nn.Dropout(0.5),

            nn.Linear(int(256), int(256)),
            nn.BatchNorm1d(int(256)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

>>>>>>> 667e376d65e10a8618272c1b1f47f6cb81ed789d
            nn.Linear(int(256), int(256)), 
            nn.BatchNorm1d(int(256)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(int(256), int(128)), 
            nn.BatchNorm1d(int(128)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )


    def forward(self, x):
        batch_size, feature_size, dim = x['face-landmarks'].size()
        x = x['face-landmarks'].view(batch_size, -1).float().to(self.device)

        x = self.feature_model(x)
        return x