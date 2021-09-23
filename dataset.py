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

import torchvision.transforms as T

from PIL import Image


#Derived from this discussion: https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/22
#Same old dataset
class HDF5Dataset(Dataset):
    """HDF5Dataset dataset. \
        author: Dusan Svilarkovic
    """

    def __init__(self, hdf_path, transform=None, use_colour=True, data_format='NCHW', preprocessing=False):
        """
        Args:
            hdf_path (string): Path to the hdf5 file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            use_colour (boolean) : To determine whether dataset has colours or not
            data_format (string) : Type of format used, default is NCHW
            preprocessing (boolean) : Whether we should normalize images or not, usually used during the training
        """

        self.hdf_path = hdf_path

        if 'test' in os.path.basename(hdf_path):
            self.landmarks = ['eye-region', 'face',
                              'face-landmarks', 'head', 'left-eye', 'right-eye']
            self.isTest = True
        else:
            self.landmarks = ['eye-region', 'face', 'face-landmarks',
                              'gaze', 'head', 'left-eye', 'right-eye']
            self.isTest = False

        self.dataset = None

        #to check if it uses color
        self._use_colour = use_colour
        #at this point useless
        self.data_format = data_format

        self.transform = transform
        self.preprocessing = preprocessing

        with h5py.File(self.hdf_path, 'r', libver='latest', swmr=True) as dataset_file:
            self.dataset = dataset_file
            self.person_keys = list(self.dataset.keys())
            ## number of persons, should be 40
            self.person_len = len(self.dataset)
            ## number of data samples per person, should be 100
            if self.isTest:
                ## First 50 person from eyeGaze--> 100 images per person
                ## Next 15 person from MPII--> 500 images per person
                self.samples_per_person_len = [100] * 50 + [500] * 15
            else:
                self.samples_per_person_len = self.dataset.get(self.person_keys[0])[
                    'face'].shape[0]

    def __len__(self):
        if self.isTest:
            return sum(self.samples_per_person_len)
        else:
            with h5py.File(self.hdf_path, 'r', libver='latest', swmr=True) as dataset_file:
                self.dataset = dataset_file
                return self.person_len * self.samples_per_person_len

    def __getitem__(self, idx):
        with h5py.File(self.hdf_path, 'r', libver='latest', swmr=True) as dataset_file:
            self.dataset = dataset_file

            if self.isTest:
                if idx < 5000:  # First 50 person from eyeGaze--> 100 images per person
                    person_id = int(idx / 100)
                    sample_num = int(idx % 100)
                else:  # Next 15 person from MPII--> 500 images per person
                    mpii_idx = idx - 5000
                    person_id = int(mpii_idx / 500) + 50
                    sample_num = int(mpii_idx % 500)
            else:
                person_id = int(idx / self.samples_per_person_len)
                sample_num = int(idx % self.samples_per_person_len)

            sample = {}

            for landmark in self.landmarks:
                sample[landmark] = self.dataset.get(self.person_keys[person_id])[
                    landmark][sample_num, ]

            if self.preprocessing:
                sample = self.preprocess_entry(sample)
            ## additional metadata, you might need it, might not
            sample['person_id'] = person_id
            sample['sample_num'] = sample_num

            sample['face'] = T.ToTensor()(Image.fromarray(sample['face']))

            if self.transform:
                sample["left-eye"] = self.transform(
                    Image.fromarray(np.uint8(sample["left-eye"])))
                sample["right-eye"] = self.transform(
                    Image.fromarray(np.uint8(sample["right-eye"])))

            return sample
    
    #singleton implementation, in order to avoid opening same dataset again and again
    #deprecated
    def set_dataset_instance(self):
        return NotImplemented

    def preprocess_entry(self, entry):
        """Normalize image intensities."""
        for k, v in entry.items():
            if v.ndim == 3:  # We get histogram-normalized BGR inputs
                if not self._use_colour:
                    v = cv.cvtColor(v, cv.COLOR_BGR2GRAY)
                v = v.astype(np.float32)
                v *= 2.0 / 255.0
                v -= 1.0
                if self._use_colour and self.data_format == 'NCHW':
                    v = np.transpose(v, [2, 0, 1])
                elif not self._use_colour:
                    v = np.expand_dims(v, axis=0 if self.data_format == 'NCHW' else -1)
                entry[k] = v

        # Ensure all values in an entry are 4-byte floating point numbers
        for key, value in entry.items():
            entry[key] = value.astype(np.float32)

        return entry
