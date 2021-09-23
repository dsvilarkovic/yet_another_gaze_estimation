import os
import running
from scripts.make_submit_file import make_test_submission
from experiment import Experiment
from experiment_deep_pictorial_gaze import ExperimentDeepPictorialGaze
from experiment_eye_for_everyone import ExperimentEyeForEveryone
from experiment_asymmetric_paper import ExperimentAsymmetricPaper
from experiment_nikola_comb import ExperimentNikolaComb
from experiment_asymmetric_paper_simplified import ExperimentAsymmetricPaperSimplified
import torchvision.transforms as T
import torch


RGB_MEAM = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)



# config = {
#     "lr" : 0.0001,
#     "wd" : 0.0005,
#     "epochs" : 60,
#     "early_stop" : 14,
#     "dataset_dir" : "/cluster/project/infk/hilliges/lectures/mp20/project3/",
#     "b_size" : 32,
#     "n_workers" : 8,
#     "save_output_dir" : None,
#     "model_name" : "Asymmetric_resnet18",
#     "debug_mode" : False,
#     "messy_param_only_final_loss" : False,
#     "code_dir" : "/cluster/scratch/dsvilarko/Cyclopes",
#     "comment" : "Original Nikola's model"
# }

config = {
    "lr" : 0.0001,
    "wd" : 0.005,
    "epochs" : 100,
    "augmentation" : True,
    "early_stop" : 14,
    "dataset_dir" : "/cluster/project/infk/hilliges/lectures/mp20/project3/",
    "b_size" : 32,
    "n_workers" : 8,
    "save_output_dir" : None,
    "model_name" : "AsymmetricUpdated_densenet161",
    "debug_mode" : False,
    "messy_param_only_final_loss" : False,
    "code_dir" : "/cluster/scratch/dsvilarko/Cyclopes"
}

#In created folder under Output folder, setup the Checkpoint file you want to separately train on 

checkpoint_file="Checkpoint_ver_2020-08-12_07-26-51.5315121597210011531"
chk_pt_dir = os.path.join(*["..", "Output", checkpoint_file])


# config["save_output_dir"] = os.path.join( *["..", "Output", "Checkpoint_ver_2020-07-01_09-47-07_best_ver_Dushan"] )


train_transform = T.Compose([T.RandomResizedCrop((60, 90), scale=(0.95, 1.05), ratio=(1.0, 1.0), interpolation=2),
                                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                T.RandomGrayscale(p=0.1),
                                T.ToTensor(),
                                T.Normalize(mean=RGB_MEAM, std=RGB_STD)])

val_transform = T.Compose([T.ToTensor(),
                            T.Normalize(mean=RGB_MEAM, std=RGB_STD)])

# #delet this
# config["use_gpu"] = False


modelName = config['model_name']

if 'EyeForEveryOne' in modelName:
    experiment1 = ExperimentEyeForEveryone(config = config, train_transform=train_transform, val_transform=val_transform)
elif "DeepPictorialGaze" in modelName:
    experiment1 = ExperimentDeepPictorialGaze(config = config, train_transform=train_transform, val_transform=val_transform)
elif "Simplified_AsymmetricUpdated" in modelName:
    # Maybe start the backbone networks from scratch
    experiment1 = ExperimentAsymmetricPaperSimplified(config = config, train_transform=train_transform, val_transform=val_transform)
elif "Asymmetric" in modelName:
    # Maybe start the backbone networks from scratch
    experiment1 = ExperimentAsymmetricPaper(config = config, train_transform=train_transform, val_transform=val_transform)
elif "NikolaComb" in modelName:
    experiment1 = ExperimentNikolaComb(config = config, train_transform=train_transform, val_transform=val_transform)
    
experiment1.checkpoint_dir = chk_pt_dir

# make_test_submission(config, experiment1, override_test_data_dir = "/cluster/scratch/dsvilarko/data/", last_or_best = 'best')
make_test_submission(config, experiment1, override_test_data_dir = "/cluster/scratch/dsvilarko/data/", last_or_best = 'best')