# from model import MultiModalUnit
# from first_model_best_0_57 import MultiModalUnit
# from dataset import HDF5Dataset
import os
import torch
from torch.utils.data import DataLoader
import numpy as np


def make_test_submission(config, experiment1, override_test_data_dir = None, last_or_best = 'best'):
    """
    custom_test_data_dir = None --> default data location
    """

    if override_test_data_dir is None:
        test_data_dir = experiment1.config['dataset_dir']
    else:
        test_data_dir = override_test_data_dir
        
    chk_pt_dir = experiment1.checkpoint_dir


    if last_or_best == 'best':
        checkpoint_file_name = os.path.join(chk_pt_dir, "best_checkpoint.pth")
    elif last_or_best == 'last':
        checkpoint_file_name = os.path.join(chk_pt_dir, "checkpoint.pth")
    else:
        print("Error!!! Invalid value passed for <<last_or_best>>.")
        return 0

    assert os.path.isdir(chk_pt_dir)
    assert os.path.isfile(checkpoint_file_name)


    submission_file_name = checkpoint_file_name[:-4] + "_submit.txt.gz"
    # probability_left_file_name = checkpoint_file_name[:-4] + "_probability_left.txt.gz"
    print('Saving prediction to :', submission_file_name)
    # print('Saving probability of E_net prediction to :', probability_left_file_name)

    # model, dev = create_model(modelName, dev = "cuda")
    model = experiment1.model
    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'

    checkpoint = torch.load(checkpoint_file_name)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # model = model.to('cuda')

    gaze_pred = []

    p_l_array = []

    test_loader = experiment1._make_dataloader(which = 'test') 
    print('>> Dataset length :', test_loader.__len__())

    for i, sample in enumerate(test_loader):

        for key in sample.keys():
            sample[key] = sample[key].to(dev) 
        
        inputs_i = sample
        print(i)
        if "DeepPictorial" in config["model_name"]:
            gaze_pred.extend(model(inputs_i)[-1].tolist())
        elif "Asymmetric" in config["model_name"]:
            gaze_pred.extend(model(inputs_i)["final-pred"].tolist())
            # p_l_array.extend(model.e_net(left_eye_image=inputs_i['left-eye'], right_eye_image=inputs_i['right-eye'])["p-l"])
        elif "NikolaComb" in config["model_name"]:
            gaze_pred.extend(model(inputs_i)["final-pred"].tolist())
        else:
            gaze_pred.extend(model(inputs_i).tolist())

    np.savetxt(submission_file_name, np.array(gaze_pred))
    # np.savetxt(probability_left_file_name, np.array(p_l_array))

