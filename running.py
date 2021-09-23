from experiment import Experiment
from experiment_deep_pictorial_gaze import ExperimentDeepPictorialGaze
from experiment_eye_for_everyone import ExperimentEyeForEveryone
from experiment_asymmetric_paper import ExperimentAsymmetricPaper
from experiment_asymmetric_paper_simplified import ExperimentAsymmetricPaperSimplified
from experiment_nikola_comb import ExperimentNikolaComb
import torchvision.transforms as T
import torch
import PIL

RGB_MEAM = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

from scripts.make_submit_file import make_test_submission

# config = {
#     "lr" : 0.001,
#     "wd" : 0.0001,
#     "epochs" : 120,  # 120,
#     "early_stop" : 25,
#     "dataset_dir" : "/scratch_net/snapo_second/nipopovic/workspace/mp_project/data", #"/cluster/project/infk/hilliges/lectures/mp20/project3/",   # datasets_dir = "/cluster/project/infk/hilliges/lectures/mp20/project3/"
#     "b_size" : 32,
#     "n_workers" : 1,
#     "use_gpu" : False,
#     "save_output_dir": "/scratch_net/snapo_second/nipopovic/workspace/mp_project/Output",  ## passing None uses Default Folder Location
#     "model_name": "DeepPictorialGaze_True",
#     "debug_mode": False,
#     "messy_param_only_final_loss": False,
# }

config = {
    "lr" : 0.0001, 
    "wd" : 0.005, 
    "epochs" : 80,  # 120,
    "early_stop" : 14,
    "dataset_dir" : "/scratch_net/snapo_second/nipopovic/workspace/mp_project/data", #"/scratch_net/snapo_second/nipopovic/workspace/mp_project/data", #"/cluster/project/infk/hilliges/lectures/mp20/project3/",   # datasets_dir = "/cluster/project/infk/hilliges/lectures/mp20/project3/"
    "b_size" : 32,
    "n_workers" : 8,
    "save_output_dir": "/scratch_net/snapo_second/nipopovic/workspace/mp_project/Output", #,  ## passing None uses Default Folder Location
    "model_name": "Simplified_AsymmetricUpdated_densenet161",   
    "debug_mode": True,
    "messy_param_only_final_loss": False,
    "augmentation": False,
    "code_dir" : "/cluster/scratch/dsvilarko/Cyclopes"
}


# TODO Newst (options are not mutually exclusive)
# TODO Option 0 - Try different eye net backbones (only 2,3 architectures before moving to 1, and them we can come back to this after 1 or 2)
# TODO Option 1 - Play arround with the loss implementation
# TODO Option 2 - Turn augmentation on/off
# TODO Option 3 (needs to be discussed) - The network has two seperate channels for two eyes....maybe combine them before the final prediction, just like in the asymmetric paper
# TODO Option 4 - For the E_net use the BaseNet in both Asymmetric and ther new proposed idea
# TODO Option 5 - Go back to Asymmetric net and modify loss (old comment: remove the weighting in the AR_net loss between two terms)
# TODO Option x ?? MAYBE??? - No image net pre training for eye networks 



if config["debug_mode"]:
    config["n_workers"] = 1

# modelName naming rule for deep pictorial gaze:
# DeepPictorialGaze_True_U_1  (either U_1 or H_numStacks)


def main():

    if config["augmentation"]:
        train_transform = T.Compose([T.RandomAffine(degrees=0.0, translate=(0.05, 0.05), scale=(0.92, 1.08), shear=None, resample=PIL.Image.BILINEAR, fillcolor=0),
                                    T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                                    T.ToTensor(),
                                    T.Normalize(mean=RGB_MEAM, std=RGB_STD),
                                    T.RandomErasing(p=0.5, scale=(0.02, 0.06), ratio=(0.3, 3.3), value=0, inplace=False)])
    else:
        train_transform = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=RGB_MEAM, std=RGB_STD)])

    val_transform = T.Compose([T.ToTensor(),
                               T.Normalize(mean=RGB_MEAM, std=RGB_STD)])

    # train_transform = T.Compose([T.ToTensor()])
    # val_transform = T.Compose([T.ToTensor()])
    # experiment1 = Experiment(config = config, train_transform=train_transform, val_transform=val_transform)
    # experiment1 = ExperimentEyeForEveryone(config = config, train_transform=train_transform, val_transform=val_transform)

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
    experiment1.train_val_test()
    
    make_test_submission(config, experiment1, override_test_data_dir = None, last_or_best = 'best')

    # make_test_submission(modelName, chk_pt_dir, batch_size = 16, last_or_best = 'best')
    
if __name__ == "__main__":
    main()