#from model import MultiModalUnit

from dataset import HDF5Dataset
from torch.utils.tensorboard import SummaryWriter
import os
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
import os
import math

import sys

from util import createDirectory, createCheckpointDir, copyToZip
from util import angular_error, mse_error


from models.first_model_best_0_57 import MultiModalUnit as EyeForEveryOne
#from models.first_model_best_0_57_improved import MultiModalUnit as EyeForEveryOne_Improved
from models.deep_pictorial_model import MultiModalUnit as DeepPictorialGaze
from models.ar_net_paper import MultiModalUnit as AsymmetricNet
from models.nikola_comb import MultiModalUnit as NikolaComb
from models.ar_net_paper_with_landmarks_and_base_net_add import MultiModalUnit as AsymmetricNetUpdated
from models.ar_net_paper_with_landmarks_and_base_net_add_simplified import MultiModalUnit as AsymmetricNetUpdated_simplified
from scripts.create_gazemap import from_gaze2d

class Experiment:
    """Comments...."""
    def __init__(self, config, train_transform, val_transform):
        self.train_transform = train_transform
        self.val_transform = val_transform

        # Load the configuration
        self.config = config       
        self._check_config_parameters()
        
        
        ## is the instance for Debugging
        self.debug = config['debug_mode']
        if(self.debug):
            config["epochs"] = 1
      
        # Directory where model checkpoints will be saved
        if self.config["save_output_dir"] is not None:
            output_folder_path = self.config["save_output_dir"]
        else:
            output_folder_path = '../Output/'
        self.checkpoint_dir = createCheckpointDir(outputFolderPath=output_folder_path, debug_mode = self.debug)
                    
        
        # copyToZip("/scratch_net/snapo_second/nipopovic/workspace/mp_project/Cyclopes/", os.path.join(self.checkpoint_dir, 'Cyclopes.zip') )
        if(config["code_dir"] is not None):
            copyToZip(config["code_dir"], os.path.join(self.checkpoint_dir, 'Cyclopes.zip'))
        
        
        # Open file for logging
        self.fh = open(os.path.join(self.checkpoint_dir, "log_file.txt"), "a")

        # Create a tensorboard summary writter
        self.summary_writer = self.create_summary_writer()

        # Create the specified model and the optimizer
        self.model, self.dev = self.create_model(self.config["model_name"])  
        self.optimizer, self.lr_scheduler = self.create_optimizer() 

        _ = random_gpu_operation()

        # Create DataLoaders for the training and validation set
        self.train_loader = self._make_dataloader(which="train") 
        self.val_loader = self._make_dataloader(which="val") 
        self.test_loader = self._make_dataloader(which="test") 
        _ = random_gpu_operation()

        # Creates the loss function and the accuracy metric
        self.loss_fn = self.create_loss_fn()

        # # Initialize epoch number
        # self.curr_epoch = 1
        # self.best_epoch = 1

        # # Initialize lists which store losses and metrics during training
        # self.train_metric = []
        # self.train_loss = []
        # self.val_metric= [] 
        # self.val_loss = []

        # TODO Enable loading
        # Try to load a pre-trained model.
        self.curr_epoch, self.best_epoch, self.val_metric, self.val_loss, self.train_metric, self.train_loss = self.try_loading(load_file=None)
        
        print('------------------------------------------')
        self.print_to_log('------------------------------------------')
        
        for keys,values in self.config.items():
            self.print_to_log( str(keys) + " : " +  str(values) )
            print( str(keys) + " : " +  str(values) )
        
        print('------------------------------------------')
        self.print_to_log('------------------------------------------')
               
    # def callModel(self, modelName):
    #     if modelName == 'EyeForEveryOne_Resnet18':
    #         print("here")
    #         from models.first_model_best_0_57 import MultiModalUnitR
        
    def _check_config_parameters(self):
        if not isinstance(self.config["lr"], float):
            raise ValueError
        elif not isinstance(self.config["wd"], float):
            raise ValueError
        elif not isinstance(self.config["b_size"], int):
            raise ValueError
        elif not isinstance(self.config["epochs"], int):
            raise ValueError
        elif not isinstance(self.config["early_stop"], int):
            raise ValueError
        elif not isinstance(self.config["n_workers"], int):
            raise ValueError
        elif not isinstance(self.config["dataset_dir"], str):
            raise ValueError
        elif not (isinstance(self.config["save_output_dir"], str) or self.config["save_output_dir"] is None):
            raise ValueError

    def create_model(self, modelName):
        if torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
        
        # callModel(modelName)
        if 'EyeForEveryOne' in modelName:
            backbone = modelName.split("_")[-1].lower()
            pretrained = False if "not_pretrained" in modelName else True
            if 'improved' in modelName:
                return EyeForEveryOne_Improved(sample_dictionary = None, feature_size=66, output_size=2, backbone = backbone, device=dev).to(dev), dev  # TODO Input/output size
            else:
                return EyeForEveryOne(sample_dictionary = None, feature_size=66, output_size=2, backbone = backbone, device=dev).to(dev), dev  # TODO Input/output size
        elif "DeepPictorialGaze_True" in modelName:
            net_1 = modelName.split("_")[-2]
            num_stacks = int(modelName.split("_")[-1])
            return DeepPictorialGaze(sample_dictionary = None, net_1 = net_1, num_stacks=num_stacks, device=dev, multiple_intermediate_losses= True).to(dev), dev  # TODO Input/output size
        elif "DeepPictorialGaze_False" in modelName:
            net_1 = modelName.split("_")[-2]
            num_stacks = int(modelName.split("_")[-1])
            return DeepPictorialGaze(sample_dictionary = None, net_1 = net_1, num_stacks=num_stacks, device=dev, multiple_intermediate_losses= False).to(dev), dev  # TODO Input/output size
        elif "Simplified_AsymmetricUpdated" in modelName:
            backbone = modelName.split("_")[-1].lower()
            return AsymmetricNetUpdated_simplified(backbone_arch=backbone, device = dev).to(dev), dev
        elif "AsymmetricUpdated" in modelName:
            pretrained = False if "not_pretrained" in modelName else True
            backbone = modelName.split("_")[-1].lower()
            return AsymmetricNetUpdated(backbone_arch=backbone, device = dev).to(dev), dev
        elif "Asymmetric" in modelName:
            backbone = modelName.split("_")[-1].lower()
            return AsymmetricNet(backbone_arch=backbone, device = dev).to(dev), dev
        elif "NikolaComb" in modelName:
            backbone = modelName.split("_")[-1].lower()
            return NikolaComb(backbone_arch=backbone, device = dev).to(dev), dev

    def _make_dataloader(self, which, override_test_data_dir = None):
        # datasets_dir = "/cluster/project/infk/hilliges/lectures/mp20/project3/"
        if which == "train":
            hdf_path = os.path.join(self.config['dataset_dir'], "mp20_train.h5")
            shuffle = True
            transform = self.train_transform
        elif which == "val":
            hdf_path = os.path.join(self.config['dataset_dir'], "mp20_validation.h5")
            shuffle = False
            transform = self.val_transform
        elif which == "test":
            if override_test_data_dir is None:
                hdf_path = os.path.join(self.config['dataset_dir'], "mp20_test_students.h5")
            else:
                hdf_path = override_test_data_dir
            shuffle = False
            transform = self.val_transform
        else:
            raise ValueError

        hdf5_dataset = HDF5Dataset(hdf_path, transform=transform, use_colour=True, data_format='NCHW', preprocessing=False)

        return DataLoader(hdf5_dataset, batch_size=self.config["b_size"], shuffle=shuffle, num_workers=self.config["n_workers"])

    def create_loss_fn(self):
#         modelName = self.config["model_name"]
        
#         def deep_pictorial_loss_fn(gaze_map_left, gaze_map_right, gaze_direction, label_gaze):
#             # make gazemap from label_gaze
#             batch_size, n_class, height, width = gaze_map_left[-1].size()

#             ALPHA=1e-5

#             loss_left  = torch.zeros(len(gaze_map_left), dtype=torch.float32).to(self.dev).detach()
#             loss_right = torch.zeros(len(gaze_map_left), dtype=torch.float32).to(self.dev).detach()
                
#             for i in range(batch_size):

#                 #there are two heatmaps
#                 for j in range(2):
#                     gazemap_left_label = from_gaze2d(label_gaze[i,:].cpu().numpy(), output_size = (height, width), scale=1.0)
#                     gazemap_left_label = torch.tensor(gazemap_left_label, dtype=torch.float32).to(self.dev)

#                     gazemap_right_label = from_gaze2d(label_gaze[i,:].cpu().numpy(), output_size = (height, width), scale=1.0)
#                     gazemap_right_label = torch.tensor(gazemap_right_label, dtype=torch.float32).to(self.dev)
                    

#                     stack_size = len(gaze_map_left)
#                     for k in reversed(range(stack_size)):
#                         loss_left[k] += ALPHA*torch.nn.BCEWithLogitsLoss(reduction='sum')(gaze_map_left[k][i,j,:,:], gazemap_left_label[j,:,:])
#                         loss_right[k] += ALPHA*torch.nn.BCEWithLogitsLoss(reduction='sum')(gaze_map_right[k][i,j,:,:], gazemap_right_label[j,:,:])

#                         if(self.model.multiple_intermediate_losses == False):
#                             #if it is not multiple intermediate losses, just skip on it and take only the last loss value
#                             break  

#             #making loss function between label gaze and final gaze
#             loss_gaze = torch.nn.MSELoss(reduction='sum')(gaze_direction, label_gaze.type(torch.float32))

#             #total_loss = total_loss + loss_gaze
#             # total_loss = (loss_left + loss_right)/2.0 + loss_gaze
#             total_loss = (loss_left + loss_right)/2.0 + 20.0* loss_gaze
#             total_loss /= batch_size
# # 
#             loss_components = {"gaze_final": loss_gaze, "loss_left": loss_left, "loss_right": loss_right}

#             return total_loss, loss_components
#             # return  loss_left + loss_val_right + loss_gaze

#         if modelName == 'EyeForEveryOne_Resnet18':
#             return torch.nn.MSELoss(reduction='mean')
#         elif "DeepPictorialGaze" in modelName:
#             return deep_pictorial_loss_fn
        raise NotImplementedError


    def create_optimizer(self):
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["wd"], momentum = 0.9)
        
        self.print_to_log(f"\nUSING ADAM OMPTIMIZER (+clamp for loss)!!!!!\n") # TODO
        optimizer = torch.optim.Adam(self.model.parameters(),
                               lr=self.config["lr"],
                               weight_decay=self.config["wd"])

        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1)

        return optimizer, lr_scheduler

    def create_summary_writer(self):
        return SummaryWriter(log_dir=os.path.join(self.checkpoint_dir, "tensorboard_logs"))

    def train_val_test(self):
        # # To meassure average epoch processing time
        # epoch_time = AverageMeter()
        # # Time of the beginning of an epoch
        # start_time = time.time()  

        # for curr_epoch in range(1, self.config["epochs"] + 1): 
        #     # Update current epoch count
        #     self.curr_epoch = curr_epoch

        #     # Log the estimated time for finishing the experiment
        #     self._log_estimated_epoch_time(epoch_time=epoch_time, lr=self.optimizer.param_groups[0]["lr"])

        #     # Train for one epoch
        #     train_loss, train_metric, train_loss_components = self.train_epoch()

        #     # Validate model on the validation set
        #     val_loss, val_metric, val_loss_components = self.validate_epoch()

        #     # Adjust the learning rate eafter each epoch, according to the lr scheduler
        #     self.lr_scheduler.step(val_loss)

        #     if self.summary_writer is not None:
        #         # Log metrics to the tensorboard
        #         self.summary_writer.add_scalars(f"Total loss",
        #                                         {"train": train_loss, "val": val_loss},
        #                                         self.curr_epoch)
        #         self.summary_writer.add_scalars(f"Metric",
        #                                         {"train": train_metric, "val": val_metric},
        #                                         self.curr_epoch)
        #         self.summary_writer.add_scalar(f"Learning rate", self.optimizer.param_groups[0]["lr"], self.curr_epoch)
        #         if train_loss_components is not None and val_loss_components is not None:
        #             self.summary_writer.add_scalars(f"Gaze direction loss",
        #                         {"train": train_loss_components["gaze_final"], "val": val_loss_components["gaze_final"]},
        #                         self.curr_epoch)
        #             for i, zippp in enumerate(zip(train_loss_components["loss_left"], val_loss_components["loss_left"], train_loss_components["loss_right"], val_loss_components["loss_right"])):
        #                 tr_int_loss_left, val_int_loss_left, tr_int_loss_right, val_int_loss_right = zippp
        #                 self.summary_writer.add_scalars(f"[{i}]Intermediate loss",
        #                                             {"train_left": tr_int_loss_left, "val_left": val_int_loss_left,
        #                                                 "train_right": tr_int_loss_right, "val_right": val_int_loss_right},
        #                                             self.curr_epoch)

        #     # Calculate epoch time, and restart timer
        #     epoch_time.update(time.time() - start_time)
        #     start_time = time.time()

        #     # Save a checkpoint of the model after each epoch
        #     self.save_checkpoint()

        #     # Check if the training should stop due to early stopping
        #     if (self.curr_epoch - self.best_epoch) == self.config["early_stop"]: 
        #         self.print_to_log("EARLY STOPPING \n")
        #         break

        # # Log the training report 
        # self.training_end_report()

        # # TODO Think later do we want to run it on test set here or not...
        # # # Load the model with the best validation performance 
        # # self.load_best_checkpoint()

        # # Close the file for logging
        # self.fh.close()

        # # # After training has finished, evaluate the model on the train/val/test and log the results.
        # # train_loss, train_metric = self.run_one_epoch(which="train", update_weights=False)
        # # val_loss, val_metric = self.run_one_epoch(which="val", update_weights=False)

        # return
        raise NotImplementedError
    
    def train_epoch(self):
        # # Train for one epoch
        # train_loss, train_metric, loss_components = self.run_one_epoch(which="train", update_weights=True)

        # # Append the loss and acc to the list (save one value for evey epoch)
        # self.train_loss.append(train_loss)
        # self.train_metric.append(train_metric)

        # return train_loss, train_metric, loss_components
        raise NotImplementedError

    def validate_epoch(self):
        # # Validate for one epoch
        # val_loss, val_metric, loss_components = self.run_one_epoch(which="val", update_weights=False)

        # # If the current validation loss is better than from all previous epochs
        # if self.curr_epoch == 1:
        #     self.best_epoch = 1
        # elif val_loss < np.min(self.val_loss):
        #     self.best_epoch = self.curr_epoch

        # # Append the loss and acc to the list (save one value for evey epoch)
        # self.val_loss.append(val_loss)
        # self.val_metric.append(val_metric)

        # return val_loss, val_metric, loss_components
        raise NotImplementedError

    def run_one_epoch(self, which, update_weights):

        # if self.config["model_name"] == "EyeForEveryOne_Resnet18":
        #     loss_avg, metric_avg = self.run_one_epoch_everyone(which, update_weights)
        #     return loss_avg, metric_avg, None
        # elif self.config["model_name"] == "DeepPictorialGaze_True":
        #     loss_avg, metric_avg, loss_components = self.run_one_epoch_pictorial(which, update_weights)
        #     return loss_avg, metric_avg, loss_components
        # elif self.config["model_name"] == "DeepPictorialGaze_False":
        #     loss_avg, metric_avg, loss_components = self.run_one_epoch_pictorial(which, update_weights)
        #     return loss_avg, metric_avg, loss_components
        raise NotImplementedError
    
    def run_one_epoch_everyone(self, which, update_weights):

        assert isinstance(update_weights, bool)

        # Take the specified data loader
        if which == "train":
            data_loader = self.train_loader
            split_name = "Training"
        elif which == "val":
            data_loader = self.val_loader
            split_name = "Validation"
        elif which == "test":
            data_loader = self.test_loader
            split_name = "Testing"
        else:
            raise AssertionError

        # Put the model in the appropriate mode. 
        # If it should update weights, put in training mode. If not, in evaluation mode.
        if update_weights:
            self.model.train()  
        else:
            self.model.eval() 

        # For averaging batch processing times over the epoch
        batch_time = AverageMeter() 
        # For averaging data loading time over the epoch
        data_time = AverageMeter() 
        # For averaging losses over the epoch
        losses = AverageMeter()
        # For storing all (prediction, target) pairs in the epoch 
        metrics = AverageMeter()

        # Measure the beginning of the batch (and also beginiing of data loading)
        batch_start_time = time.time()  

        # Loop over the whole dataset once
        for i, sample in enumerate(data_loader):
            ## in debug mode do not train over entire sample
            if self.debug:
                print(i)
                if i == 3:
                    break
            
            # Unpack the current batch 
            for key in sample.keys():
                sample[key] = sample[key].to(self.dev) 
                
            inputs_i = sample
            labels_i = sample['gaze']
            
            # Measure data loading/pre-processing time
            data_time.update(time.time() - batch_start_time)  
            
            pred = self.model(inputs_i)

            loss = self.loss_fn(pred, labels_i.type(torch.float32)) 

            # Delete calculated gradients
            self.optimizer.zero_grad()  

            # If in weight update mode
            if update_weights:
                # Calculate the loss gradients
                loss.backward()
                # Update network weights with calculated gradients  
                self.optimizer.step()
                # Delete calculated gradients
                self.optimizer.zero_grad()   

            # Update epoch loss averaging
            losses.update(loss.item())

            # Update accuracy metric calculator
            with torch.no_grad():
                error = self.calculate_metric(pred=pred, gt=labels_i)

                # Update epoch metric averaging
                metrics.update(error.item())
                
            
            del sample, inputs_i, labels_i, pred, loss, error
            torch.cuda.empty_cache()

            # Measure the time it took to process the batch
            batch_time.update(time.time() - batch_start_time)
            # Measure the beginning of the next ba
            batch_start_time = time.time()  

        # Calculate average epoch loss
        loss_avg = losses.get_average()
        # Calculate the average of the metric on this epoch
        metric_avg = metrics.get_average()

        self.print_to_log(f"{split_name} loss: {loss_avg:.6f}")
        self.print_to_log(f"{split_name} metric: {metric_avg}")
        
        return loss_avg, metric_avg

    def run_one_epoch_pictorial(self, which, update_weights):

        assert isinstance(update_weights, bool)

        # Take the specified data loader
        if which == "train":
            data_loader = self.train_loader
            split_name = "Training"
        elif which == "val":
            data_loader = self.val_loader
            split_name = "Validation"
        elif which == "test":
            data_loader = self.test_loader
            split_name = "Testing"
        else:
            raise AssertionError

        # Put the model in the appropriate mode. 
        # If it should update weights, put in training mode. If not, in evaluation mode.
        if update_weights:
            self.model.train()  
        else:
            self.model.eval() 

        # For averaging batch processing times over the epoch
        batch_time = AverageMeter() 
        # For averaging data loading time over the epoch
        data_time = AverageMeter() 
        # For averaging losses over the epoch
        losses = AverageMeter()
        # For storing all (prediction, target) pairs in the epoch 
        metrics = AverageMeter()

        # Measure the beginning of the batch (and also beginiing of data loading)
        batch_start_time = time.time()  

        loss_components = {}

        # Loop over the whole dataset once
        for i, sample in enumerate(data_loader):
            
            
            ## in debug mode do not train over entire sample
            if self.debug:
                print(i)
                if i == 3:
                    break
            
            # Unpack the current batch 
            for key in sample.keys():
                sample[key] = sample[key].to(self.dev) 
                
            inputs_i = sample
            labels_i = sample['gaze']
            
            # Measure data loading/pre-processing time
            data_time.update(time.time() - batch_start_time)  

            # Compute network raw prediction
            # pred = self.model(inputs_i)  
            # TODO
            
            gaze_map_left, gaze_map_right,  pred = self.model(inputs_i)
            
            # x = self.model(inputs_i)
            # print(x)
            # gaze_map_left, gaze_map_right,  gaze_direction = x
            

            # Compute the loss 
            # loss = self.loss_fn(pred, labels_i) 
                # TODO 
            loss, loss_cmp = self.loss_fn(gaze_map_left, gaze_map_right, pred, labels_i) 

            if i == 0 :
                loss_components["gaze_final"] = loss_cmp["gaze_final"].clone().detach()
                loss_components["loss_left"] = loss_cmp["loss_left"].clone().detach()
                loss_components["loss_right"] = loss_cmp["loss_right"].clone().detach()
            else:
                loss_components["gaze_final"] += loss_cmp["gaze_final"].clone().detach()
                loss_components["loss_left"] += loss_cmp["loss_left"].clone().detach()
                loss_components["loss_right"] += loss_cmp["loss_right"].clone().detach()


            # TODO DELETE!!!!!!!!!!!
            if self.config["messy_param_only_final_loss"]:
                loss = loss_cmp["gaze_final"]

            # Delete calculated gradients
            self.optimizer.zero_grad()  

            # If in weight update mode
            if update_weights:
                # Calculate the loss gradients
                loss.backward()
                # Update network weights with calculated gradients  
                self.optimizer.step()
                # Delete calculated gradients
                self.optimizer.zero_grad()   

            # Update epoch loss averaging
            losses.update(loss.item())

            # Update accuracy metric calculator
            with torch.no_grad():
                error = self.calculate_metric(pred=pred, gt=labels_i)

                # Update epoch metric averaging
                metrics.update(error.item())
                
            
            del sample, inputs_i, labels_i, gaze_map_left, gaze_map_right, pred, loss, loss_cmp, error
            torch.cuda.empty_cache()

            # Measure the time it took to process the batch
            batch_time.update(time.time() - batch_start_time)
            # Measure the beginning of the next ba
            batch_start_time = time.time()  

        # Calculate average epoch loss
        loss_avg = losses.get_average()
        # Calculate the average of the metric on this epoch
        metric_avg = metrics.get_average()
        # Calculate the average of the loss components
        loss_components["gaze_final"] /= (i+1)
        loss_components["loss_left"] /= (i+1)
        loss_components["loss_right"] /= (i+1)

        self.print_to_log(f"{split_name} loss: {loss_avg:.6f}")
        self.print_to_log(f"{split_name} metric: {metric_avg}")
        
        return loss_avg, metric_avg, loss_components

    def calculate_metric(self, pred, gt):
        return angular_error(pred, gt)

    def try_loading(self, load_file=None):
        """
        If an ongoing experiment is being continued, load the model. Otherwise, don't load.

        :return: curr_epoch, train_acc, train_loss, best_epoch, val_acc, val_loss
        """
        curr_epoch = 1
        best_epoch = 1
        val_metric = []
        val_loss = []
        train_metric = []
        train_loss = []

        if load_file is not None:

            # Load the checkpoint
            curr_epoch, best_epoch, val_metric, val_loss, train_metric, train_loss = self.load_checkpoint(checkpoint_path=load_file, just_metrics=False)

            # If a model is loaded at epoch x, that means that the model trained for x epochs.
            # Because of that, we want to start fresh from epoch x+1,
            curr_epoch = curr_epoch + 1

        return curr_epoch, best_epoch, val_metric, val_loss, train_metric, train_loss

    def save_checkpoint(self):
        self.model.eval()  # Switch the model to evaluation mode
        state = {'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'curr_epoch': self.curr_epoch,
                 'best_epoch': self.best_epoch,
                 'train_metric': self.train_metric,
                 'train_loss': self.train_loss,
                 'val_metric': self.val_metric,
                 'val_loss': self.val_loss}
        torch.save(state, os.path.join(self.checkpoint_dir, "checkpoint.pth"))
        if self.curr_epoch == self.best_epoch:
            torch.save(state, os.path.join(self.checkpoint_dir, "best_checkpoint.pth"))

        return

    def load_best_checkpoint(self):
        """
        Loads the best model
        """
        best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pth")
        _, _, _, _, _, _ = self.load_checkpoint(checkpoint_path=best_path, just_metrics=False)
    
    def print_to_log(self, message):
        print(message, file=self.fh)
        self.fh.flush()

    def load_checkpoint(self, checkpoint_path, just_metrics=False):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        curr_epoch = checkpoint['curr_epoch']
        best_epoch = checkpoint['best_epoch']
        val_metric = checkpoint['val_metric']
        val_loss = checkpoint['val_loss']
        train_metric = checkpoint['train_metric']
        train_loss = checkpoint['train_loss']

        if not just_metrics:
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.print_to_log("Loaded checkpoint model on CPU")
            self.print_to_log("Checkpoint found at: '{}'".format(checkpoint_path))
            self.print_to_log(f"The loaded model has trained up to epoch:{curr_epoch},"
                      f" with the validation loss: {val_loss[-1]:.4f},"
                      f" and metric value: {val_metric[-1]:.4f}.")
            self.print_to_log(" ")

        return curr_epoch, best_epoch, val_metric, val_loss, train_metric, train_loss

    def training_end_report(self):
        # Best epoch indexes to extract best metrics from arrays
        best_tr_epoch_i = int(np.argmin(self.train_loss))
        best_val_epoc_i = self.best_epoch - 1  # Minus one because epochs start from 1, and list indexing starts from 0

        self.print_to_log(" ")
        self.print_to_log("---------------------------------------------------------------------------------")
        self.print_to_log("End of training report:")
        self.print_to_log("---------------------------------------------------------------------------------")
        self.print_to_log("Best training loss: {:0.4f}".format(self.train_loss[best_tr_epoch_i]))
        # self.print_to_log("Best training metric: {:0.4f}".format(self.train_metric[best_tr_epoch_i]))
        self.print_to_log("Best validation loss: {:0.4f}".format(self.val_loss[best_val_epoc_i]))
        # self.print_to_log("Best validation metric: {:0.4f}".format(self.val_metric[best_val_epoc_i]))
        self.print_to_log("Epoch with the best training loss: {}".format(best_tr_epoch_i + 1))
        self.print_to_log("Epoch with the best validation loss: {}".format(self.best_epoch))
        self.print_to_log("---------------------------------------------------------------------------------")
        self.print_to_log(" ")
        self.print_to_log("Finished training. Starting the evaluation.")
        self.print_to_log("---------------------------------------------------------------------------------")
        self.print_to_log(" ")
        return
    
    def _log_estimated_epoch_time(self, epoch_time, lr):
        """
        self.Print_to_logself.print_to_log the estimated time to finish this experiment, as well as the lr for the current epoch.

        :param epoch_time:      average time per one epoch
        :param lr:              current lr
        """
        # Info about the last epoch
        # (Do not self.print_to_log before first epoch happens)
        if epoch_time.val != 0.0:

            epoch_h, epoch_m, epoch_s = convert_secs2time(epoch_time.val)
            self.print_to_log('Epoch processing time: {:02d}:{:02d}:{:02d} (H:M:S) \n'.format(epoch_h, epoch_m, epoch_s))

        # Info about the beginning of the current epoch
        remaining_seconds = epoch_time.get_average() * (self.config["epochs"] - self.curr_epoch)  
        need_hour, need_mins, _ = convert_secs2time(remaining_seconds)
        need_time = '[Need: {:02d}:{:02d} (H:M)]'.format(need_hour, need_mins)
        self.print_to_log('{:3d}/{:3d} ----- [{:s}] {:s} LR={:}'.format(self.curr_epoch, self.config["epochs"], time_string(), need_time, lr))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def get_average(self):
        return self.avg

def random_gpu_operation():
    """
    Does some operation on the GPU, in order for the experiment not to go IDLE before everything is loaded prior
    to the start of training
    """
    if torch.cuda.is_available():
        dev = "cuda"
        print("Random operation being done on device: {} \n".format(dev))
        t1 = torch.rand((300, 300), dtype=torch.float64)
        t1 = t1.to(dev)
        t2 = torch.rand((300, 300), dtype=torch.float64)
        t2 = t2.to(dev)
        t3 = t2 * t1
        return t3
    else:
        return None

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs

def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
    return string


 