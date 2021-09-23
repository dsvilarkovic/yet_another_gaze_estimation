from experiment import Experiment
from experiment_deep_pictorial_gaze import ExperimentDeepPictorialGaze
from experiment_eye_for_everyone import ExperimentEyeForEveryone
from experiment_asymmetric_paper import ExperimentAsymmetricPaper
from experiment_nikola_comb import ExperimentNikolaComb
import torchvision.transforms as T
import torch
import sys
import json
import PIL

RGB_MEAM = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

from scripts.make_submit_file import make_test_submission

def main():
    config = None
    with open(sys.argv[1]) as json_file:
        config = json.load(json_file)

    #for debugging
    if(len(sys.argv) > 2):
        config["debug_mode"] = True if "debug" in sys.argv[2] else False


    assert(config is not None)
    # config["code_dir"] =  "/cluster/scratch/dsvilarko/Cyclopes" 

    if "augmentation" in config.keys() and config["augmentation"]:
        # train_transform = T.Compose([T.RandomAffine(degrees=0.0, translate=(0.05, 0.05), scale=(0.92, 1.08), shear=None, resample=PIL.Image.BILINEAR, fillcolor=0),
        #                             T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        #                             T.ToTensor(),
        #                             T.Normalize(mean=RGB_MEAM, std=RGB_STD),
        #                             T.RandomErasing(p=0.5, scale=(0.02, 0.06), ratio=(0.3, 3.3), value=0, inplace=False)])
        train_transform = T.Compose([T.RandomAffine(degrees=0.0, translate=(0.1, 0.1), scale=(0.7, 1.3), shear=None, resample=PIL.Image.BILINEAR, fillcolor=0),
                            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                            T.ToTensor(),
                            T.Normalize(mean=RGB_MEAM, std=RGB_STD),
                            T.RandomErasing(p=0.5, scale=(0.1, 0.1), ratio=(0.4, 3.4), value=0, inplace=False)])
        # train_transform = T.Compose([T.RandomAffine(degrees=0.0, translate=(0.2, 0.2), scale=(0.5, 1.5), shear=None, resample=PIL.Image.BILINEAR, fillcolor=0),
        #                     T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        #                     T.ToTensor(),
        #                     T.Normalize(mean=RGB_MEAM, std=RGB_STD),
        #                     T.RandomErasing(p=0.5, scale=(0.2, 0.2), ratio=(0.4, 3.4), value=0, inplace=False)])
    else:
        train_transform = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=RGB_MEAM, std=RGB_STD)])

    val_transform = T.Compose([T.ToTensor(),
                               T.Normalize(mean=RGB_MEAM, std=RGB_STD)])

    modelName = config['model_name']
    if 'EyeForEveryOne' in modelName:
        experiment1 = ExperimentEyeForEveryone(config = config, train_transform=train_transform, val_transform=val_transform)
    elif "DeepPictorialGaze" in modelName:
        experiment1 = ExperimentDeepPictorialGaze(config = config, train_transform=train_transform, val_transform=val_transform)
    elif "Asymmetric" in modelName:
        # Maybe start the backbone networks from scratch
        experiment1 = ExperimentAsymmetricPaper(config = config, train_transform=train_transform, val_transform=val_transform)
    elif "NikolaComb" in modelName:
        experiment1 = ExperimentNikolaComb(config = config, train_transform=train_transform, val_transform=val_transform)
        
    experiment1.train_val_test()
    
    
    # make_test_submission(config, experiment1, override_test_data_dir = '/cluster/scratch/dsvilarko/data/', last_or_best = 'best')
    make_test_submission(config, experiment1, override_test_data_dir = None, last_or_best = 'best')

    
if __name__ == "__main__":
    main()