

import os
import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset_debugging import HDF5Dataset

from models.first_model_best_0_57 import MultiModalUnit as EyeForEveryOne
from models.deep_pictorial_model import MultiModalUnit as DeepPictorialGaze
from models.ar_net_paper import MultiModalUnit as AsymmetricNet
from scripts.create_gazemap import from_gaze2d

import torchvision.transforms as T
import torch
import PIL


RGB_MEAM = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


test_data_dir = "/scratch_net/snapo_second/nipopovic/workspace/mp_project/data"
modelName = "Asymmetric_resnet18"
chk_pt_dir = "/scratch_net/snapo_second/nipopovic/workspace/mp_project/Output/Checkpoint_ver_2020-07-22_16-50-15"
last_or_best = 'best'
dev = "cpu"
b_size = 64
n_workers = 1

hdf_path_list = [   
                    os.path.join(test_data_dir, "m20_test_gazeCapture.h5"),
                    os.path.join(test_data_dir, "mp20_test_MPII.h5"),
                ]

for f in hdf_path_list:
    assert os.path.isfile(f)

if last_or_best == 'best':
    checkpoint_file_name = os.path.join(chk_pt_dir, "best_checkpoint.pth")
elif last_or_best == 'last':
    checkpoint_file_name = os.path.join(chk_pt_dir, "checkpoint.pth")
else:
    print("Error!!! Invalid value passed for <<last_or_best>>.")
    exit()

assert os.path.isdir(chk_pt_dir)
assert os.path.isfile(checkpoint_file_name)

for f in hdf_path_list:
    assert os.path.isfile(f)

submission_file_name = checkpoint_file_name[:-4] + "_submit.txt.gz"
print('Saving prediction to :', submission_file_name)

if 'EyeForEveryOne' in modelName:
    backbone = modelName.split("_")[-1].lower()
    pretrained = False if "not_pretrained" in modelName else True
    model = EyeForEveryOne(sample_dictionary = None, feature_size=66, output_size=2, backbone = backbone, device=dev).to(dev)  # TODO Input/output size
elif "DeepPictorialGaze_True" in modelName:
    net_1 = modelName.split("_")[-2]
    num_stacks = int(modelName.split("_")[-1])
    model = DeepPictorialGaze(sample_dictionary = None, net_1 = net_1, num_stacks=num_stacks, device=dev, multiple_intermediate_losses= True).to(dev)  # TODO Input/output size
elif "DeepPictorialGaze_False" in modelName:
    net_1 = modelName.split("_")[-2]
    num_stacks = int(modelName.split("_")[-1])
    model = DeepPictorialGaze(sample_dictionary = None, net_1 = net_1, num_stacks=num_stacks, device=dev, multiple_intermediate_losses= False).to(dev)  # TODO Input/output size
elif "Asymmetric" in modelName:
    backbone = modelName.split("_")[-1].lower()
    model = AsymmetricNet(backbone_arch=backbone, device = dev).to(dev)

checkpoint = torch.load(checkpoint_file_name, map_location="cpu")

model.load_state_dict(checkpoint['state_dict'])
model.eval()

gaze_pred = []

for hdf_path in hdf_path_list:
    print('>> Predicting on :', hdf_path)

    transform_l = T.Compose([T.ToTensor(), T.Normalize(mean=RGB_MEAM, std=RGB_STD)])
    transform_l = T.Compose([T.RandomAffine(degrees=0.0, translate=(0.05, 0.05), scale=(0.92, 1.08), shear=None, resample=PIL.Image.BILINEAR, fillcolor=0),
                            T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                            T.ToTensor(),
                            T.Normalize(mean=RGB_MEAM, std=RGB_STD),
                            T.RandomErasing(p=1.0, scale=(0.02, 0.06), ratio=(0.3, 3.3), value=0, inplace=False)])
    transform_r = T.Compose([T.ToTensor(), T.Normalize(mean=RGB_MEAM, std=RGB_STD)])
    hdf5_dataset = HDF5Dataset(hdf_path, transform_l=transform_l, transform_r=transform_r, use_colour=True, data_format='NCHW', preprocessing=False)
    test_loader = DataLoader(hdf5_dataset, batch_size=1, shuffle=False, num_workers=n_workers)

    print('>> Dataset length :', test_loader.__len__())

    for i, sample in enumerate(test_loader):
        print(i)

        for key in sample.keys():
            sample[key] = sample[key].to(dev) 
        
        inputs_i = sample

        if "DeepPictorial" in modelName:
            gaze_pred.extend(model(inputs_i)[-1].tolist())
        elif "Asymmetric" in modelName:
            pred = model(inputs_i)
            p_l = pred["p-l"]
            p_r = pred["p-r"]
            g_l = pred["gaze-left"]
            g_r = pred["gaze-right"]
            final_pred = pred["final-pred"]
            print(f"p-l:{p_l.item():.3f}...p-r{p_r.item():.3f}")
            print(f"gaze-l:{g_l.tolist()}...gaze-r{g_r.tolist()}")
            print(f"final pred:{final_pred.tolist()}\n\n")
            
            gaze_pred.extend(final_pred)
        else:
            gaze_pred.extend(model(inputs_i).tolist())

np.savetxt(submission_file_name, np.array(gaze_pred))

