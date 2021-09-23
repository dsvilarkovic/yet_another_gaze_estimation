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
from models.resnet_head import resnet18, resnet34, resnet50
from  torch.nn.modules.upsampling import Upsample
from models.dense_net_eyeForEveryone import densenet121

class MultiModalUnit(nn.Module):
    """
        Landmark units to be trained on, there are 33x2 landmarks
        
        inputs:
             feature_size (int) : size of features nx2
             output_size (int) : size to be output from this model
             sample_dictionary (dictionary) : used for extracting shape sizes and setting up adequate convolutional 
             pretrainted (boolean) : used for Resnet architectures only , for DenseNet needs to be set up manually
             backbone (string) : type of backbone used for model, includes : 'resnet18','resnet34','resnet50','densenet121'.

    """
    def __init__(self, sample_dictionary, feature_size = 66, output_size = 2, backbone = 'resnet18', device = 'cuda', pretrained = True):
        super(MultiModalUnit, self).__init__()        

        self.device = device
        self.landmark_unit = LandmarkUnit(device = device).to(device)
        self.image_unit = ImageUnit(sample_dictionary, device = device, backbone = backbone).to(device)
        
        self.fc_final_f1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc_final_f2 = nn.Linear(16, 2)
        
        return 
        
    def forward(self, x):
        landmark_unit_output = self.landmark_unit(x)
        image_unit_output = self.image_unit(x)
        
        input_data = torch.cat([landmark_unit_output, image_unit_output], dim = 1)
        x = self.fc_final_f1(input_data)
        x = self.relu(x)
        x = self.fc_final_f2(x)
        
        return x
        

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
            nn.Linear(feature_size, int(feature_size/2)) , 
            nn.ReLU(inplace=True) ,
            nn.Dropout(0.5),
            
            nn.Linear(int(feature_size/2), int(feature_size/2)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(int(feature_size/2), int(feature_size/4)), 
            nn.ReLU(inplace=True) ,
            nn.Dropout(0.2),

            nn.Linear(int(feature_size/4), 2)
        )



    def forward(self, x):
        batch_size, feature_size, dim = x['face-landmarks'].size()
        x = x['face-landmarks'].view(batch_size, -1).float().to(self.device)

        x = self.feature_model(x)


        return x

#Inspiration from https://people.csail.mit.edu/khosla/papers/cvpr2016_Khosla.pdf
class ImageUnit(nn.Module):
    """
        Image unit neural network for extracting important images from dictionary. \n
        Those are: eye-region, face, left-eye, right-eye. \n
        
        inputs:
            sample_dictionary (dictionary) : used for extracting shape sizes and setting up adequate convolutional network sizes
            pretrainted (boolean) : used for Resnet architectures only , for DenseNet needs to be set up manually
            backbone (string) : type of backbone used for model, includes : 'resnet18','resnet34','resnet50','densenet121'.
    """
    def __init__(self, sample_dictionary, device = 'cuda', backbone = 'resnet18', pretrained = True):
        super(ImageUnit, self).__init__()
        
        # self.eye_region_height, self.eye_region_width, _ = sample_dictionary['eye-region'].shape
        # self.one_eye_height, self.one_eye_width, _ = sample_dictionary['left-eye'].shape
        # self.face_height, self.face_width, _ = sample_dictionary['face'].shape
        
        self.eye_region_height, self.eye_region_width, _ = (60,224,3)
        self.one_eye_height, self.one_eye_width, _ = (60,90,3)
        self.face_height, self.face_width, _ = (60,90,3)

        self.device = device
     
        #Left and right eye shared weights
        if backbone == 'resnet18':
            self.eye_region = resnet18(pretrained=pretrained, progress=False).to(device)
        elif backbone == 'resnet34':
            self.eye_region = resnet34(pretrained=pretrained, progress=False).to(device)
        elif backbone == 'resnet50':
            self.eye_region = resnet50(pretrained=pretrained, progress=False).to(device)
        elif backbone == 'densenet121':
            self.eye_region = densenet121(pretrained=False, progress=False).to(device)
        else:
            raise AssertionError

        self.eye_upsample = Upsample(scale_factor=2.5, mode='bicubic')
        
        # fc for connecting left and right eye
        if backbone == 'resnet50':
            self.fc_e1 = nn.Sequential(
                nn.Linear(2 * 6* 512 * 4, 2 * 6* 512),
                nn.ReLU(),
                nn.Linear(2 * 6* 512, 128)
            ) 
        else:   
            self.fc_e1 = nn.Linear(2 * 6* 512, 128)
        self.dropout_fc_e1 = nn.Dropout(0.5)
        

        # Face weights
        if backbone == 'resnet18':
            self.face_region = resnet18(pretrained=pretrained, progress=False).to(device)
        elif backbone == 'resnet34':
            self.face_region = resnet34(pretrained=pretrained, progress=False).to(device)
        elif backbone == 'resnet50':
            self.face_region = resnet50(pretrained=pretrained, progress=False).to(device)
        elif backbone == 'densenet121':
            self.face_region = densenet121(pretrained=False, progress=False).to(device)
        else:
            raise AssertionError


        self.face_upsample= Upsample(scale_factor=2.5, mode='bicubic')

        #paper doesn't have this file, kinda funny
        self.conv_f5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=1, padding=(1, 1))

        
        # fc for face weights
        # self.fc_f1 = nn.Linear(6*512, 128)
        if backbone == 'resnet50' : 
            self.fc_f1 = nn.Sequential(
                        nn.Linear(6 * 512 * 4, 6 * 512),
                        nn.ReLU(),
                        nn.Linear(6 * 512, 512),
                        nn.ReLU(),
                        nn.Linear(512, 128))
        else: 
            self.fc_f1 = nn.Sequential(
                        nn.Linear(6 * 512, 512),
                        nn.ReLU(),
                        nn.Linear(512, 128))
        self.fc_f2 = nn.Linear(128, 64)
        
        #finally , for the output
        
        #fc-f2 + fc-e1 + maybe fc-fg1 (input and face masks needs to be reconsidered)
        self.fc1 = nn.Linear(64 + 128, 128)
        self.fc2 = nn.Linear(128, 2)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dropout50 = nn.Dropout(0.5)


        
    def forward(self,landmark):
        batch_size, _, _, _= landmark['eye-region'].shape
        #left eye
        left_eye = self.forward_eye(landmark['left-eye'])
        
        #right eye
        right_eye = self.forward_eye(landmark['right-eye'])

        
        #left_eye + right_eye
        eyes = torch.cat([left_eye, right_eye], dim=1)
        eyes = eyes.view(batch_size, -1)
        fc_e1 = self.fc_e1(eyes)
        fc_e1 = self.relu(fc_e1)
        
        #face
        face = self.forward_face(landmark['face'])
        
        #fc1 and fc2 with face + fc_f1 connected

        face_and_eyes = torch.cat([face, fc_e1], dim = 1)
        
        output = self.fc1(face_and_eyes)
        output = self.relu(output)
        output = self.fc2(output)
        
        
        return output
    
    def forward_eye(self, x):
        """Used for taking either right-eye or left-eye for forwarding. \n
        Using shared weights as in the paper referenced"""
        
#         x = x.permute(0,3,1,2).float()
        x = x.float()
    
        x = self.eye_upsample(x)
        x = self.eye_region(x)

        return x
    
    def forward_face(self, x):
        batch_size = x.shape[0]
#         x = x.permute(0,3,1,2).float()
        batch_size = x.shape[0]
        x = self.face_region(x)
        
        x = self.fc_f1(x.view(batch_size, -1))
        x = self.relu(x)
        x = self.dropout50(x)
        
        x = self.fc_f2(x)
        x = self.relu(x)
        x = self.dropout50(x)

        
        return x
