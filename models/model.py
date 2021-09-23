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
from resnet_head import resnet18, resnet34, resnet50
from  torch.nn.modules.upsampling import Upsample

from torchsummary import summary


class MultiModalUnit(nn.Module):
    """
        Multimodal Unit is now useless, since most of the work is based on ImageUnit, where LandmarkUnit is being initialized
        
        inputs:
             feature_size (int) : size of features nx2
             output_size (int) : size to be output from this model
             sample_dictionary (dictionary) : used for extracting shape sizes and setting up adequate convolutional \n

    """
    def __init__(self, sample_dictionary, feature_size = 66, output_size = 2, device = 'cuda'):
        super(MultiModalUnit, self).__init__()        

        self.device = device
        self.image_unit = ImageUnit(sample_dictionary, device = device).to(device)
        
        return 
        
    def forward(self, x):
        image_unit_output = self.image_unit(x)        
        return image_unit_output
        

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
            # nn.BatchNorm1d(int(feature_size/2)) ,
            nn.ReLU(inplace=True) ,
            nn.Dropout(0.5),
            
            nn.Linear(int(feature_size/2), int(feature_size/2)),
            # nn.BatchNorm1d(int(feature_size/2)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(int(feature_size/2), int(feature_size/4)), 
            # nn.BatchNorm1d(int(feature_size/4)),
            nn.ReLU(inplace=True) ,
            nn.Dropout(0.2),
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
            sample_dictionary (dictionary) : used for extracting shape sizes and setting up adequate convolutional \n
            network sizes
    """
    def __init__(self, sample_dictionary, device = 'cuda'):
        super(ImageUnit, self).__init__()
        
        # self.eye_region_height, self.eye_region_width, _ = sample_dictionary['eye-region'].shape
        # self.one_eye_height, self.one_eye_width, _ = sample_dictionary['left-eye'].shape
        # self.face_height, self.face_width, _ = sample_dictionary['face'].shape
        
        self.eye_region_height, self.eye_region_width, _ = (60,224,3)
        self.one_eye_height, self.one_eye_width, _ = (60,90,3)
        self.face_height, self.face_width, _ = (60,90,3)


        self.device = device
                
        
        self.eye_upsample = Upsample(scale_factor=2.5, mode='bicubic')
        self.face_upsample= Upsample(scale_factor=2.5, mode='bicubic')
        
              
        self.eye_region = resnet18(pretrained=True, progress=False).to(device)
        self.face_region = resnet18(pretrained=True, progress=False).to(device)
        
        self.landmark_unit = LandmarkUnit().to(device)
        
        
        # fc for connecting left and right eye
        # self.fc_e1 = nn.Linear(2 * 6* 512, 128)
        self.fc_e1 = nn.Sequential(
                    nn.Linear(2 * 6* 512, 2 * 512),
                    nn.Linear(2 * 512, 128))

        self.dropout_fc_e1 = nn.Dropout(0.5)
        
  
        # fc for face weights
        # self.fc_f1 = nn.Linear(6*512, 128)
        self.fc_f1 = nn.Sequential(
                    nn.Linear(6*512, 512),
                    nn.Linear(512, 128))

        self.fc_f2 = nn.Linear(128, 64)
        
        #finally , for the output
        #fc-f2 + fc-e1 + maybe fc-fg1 (input and face masks needs to be reconsidered)
        self.fc1 = nn.Linear(64 + 128 + 16, 128)
        # self.fc2 = nn.Linear(128, 2)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)


        
        self.relu = nn.ReLU()

        self.dropout50 = nn.Dropout(0.5)


    def forward(self,landmark):
        batch_size, _, _, _= landmark['eye-region'].shape
        
        #face landmarks 
        face_landmarks = self.landmark_unit(landmark)
        
        #left eye
        left_eye = self.forward_eye(landmark['left-eye'])
        
        #right eye
        right_eye = self.forward_eye(landmark['right-eye'])

        
        #left_eye + right_eye
        eyes = torch.cat([left_eye, right_eye], dim=1)
        eyes = eyes.view(batch_size, -1)
        
        fc_e1 = self.fc_e1(eyes)
        # fc_e1 = self.batch_norm_fc_e1(fc_e1)
        fc_e1 = self.relu(fc_e1)
        fc_e1 = self.dropout50(fc_e1)
        
        #face
        face = self.forward_face(landmark['face'])
        
        #fc1 and fc2 with face + fc_f1 connected

        face_and_eyes = torch.cat([face, fc_e1, face_landmarks], dim = 1)
        
        output = self.fc1(face_and_eyes)

        output = self.relu(output)

        output = self.fc2(output)

        output = self.relu(output)
        
        output = self.fc3(output)
        return output
    
    
    def forward_eye(self, x):
        """Used for taking either right-eye or left-eye for forwarding. \n
        Using shared weights as in the paper referenced"""
        x = x.float()

        x = self.eye_upsample(x)
        x = self.eye_region(x)
        return x
    
    def forward_face(self, x):
        batch_size = x.shape[0]
        x = self.face_region(x)
        
        x = self.fc_f1(x.view(batch_size, -1))
        x = self.relu(x)
        x = self.dropout50(x)
        
        x = self.fc_f2(x)
        x = self.relu(x)
        x = self.dropout50(x)
        
        return x
    