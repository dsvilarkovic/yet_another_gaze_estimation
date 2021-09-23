#from model import MultiModalUnit

from dataset import HDF5Dataset
from torch.utils.tensorboard import SummaryWriter
import os
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
import os

import sys

from util import createDirectory, createCheckpointDir, copyToZip


from models.deep_pictorial_model import MultiModalUnit as DeepPictorialGaze
from scripts.create_gazemap import from_gaze2d
from experiment import Experiment,AverageMeter

class ExperimentDeepPictorialGaze(Experiment):
    """
        Class that inherits all the functions from the Experiment class and reimplements the ones that need to be implemented
    """
    def __init__(self, config, train_transform, val_transform):
        """
            Explicitly mentioned initiualization, by default super() is called
        """
        super().__init__(config, train_transform, val_transform)


    def _check_config_parameters(self):
        return super()._check_config_parameters()

    def create_model(self, modelName):
        return super().create_model(modelName)

    def _make_dataloader(self, which, override_test_data_dir = None):
        return super()._make_dataloader(which, override_test_data_dir)

    def create_loss_fn(self):
        modelName = self.config["model_name"]

        def deep_pictorial_loss_fn(gaze_map_left, gaze_map_right, gaze_direction, label_gaze):
            # make gazemap from label_gaze
            batch_size, n_class, height, width = gaze_map_left[-1].size()

            ALPHA=1e-5

            loss_left  = torch.zeros(len(gaze_map_left), dtype=torch.float32).to(self.dev).detach()
            loss_right = torch.zeros(len(gaze_map_left), dtype=torch.float32).to(self.dev).detach()
                
            for i in range(batch_size):

                #there are two heatmaps
                for j in range(2):
                    gazemap_left_label = from_gaze2d(label_gaze[i,:].cpu().numpy(), output_size = (height, width), scale=1.0)
                    gazemap_left_label = torch.tensor(gazemap_left_label, dtype=torch.float32).to(self.dev)

                    gazemap_right_label = from_gaze2d(label_gaze[i,:].cpu().numpy(), output_size = (height, width), scale=1.0)
                    gazemap_right_label = torch.tensor(gazemap_right_label, dtype=torch.float32).to(self.dev)
                    

                    stack_size = len(gaze_map_left)
                    for k in reversed(range(stack_size)):
                        loss_left[k] += ALPHA*torch.nn.BCEWithLogitsLoss(reduction='sum')(gaze_map_left[k][i,j,:,:], gazemap_left_label[j,:,:])
                        loss_right[k] += ALPHA*torch.nn.BCEWithLogitsLoss(reduction='sum')(gaze_map_right[k][i,j,:,:], gazemap_right_label[j,:,:])

                        if(self.model.multiple_intermediate_losses == False):
                            #if it is not multiple intermediate losses, just skip on it and take only the last loss value
                            break  

            #making loss function between label gaze and final gaze
            loss_gaze = torch.nn.MSELoss(reduction='sum')(gaze_direction, label_gaze.type(torch.float32))

            #total_loss = total_loss + loss_gaze
            # total_loss = (loss_left + loss_right)/2.0 + loss_gaze
            total_loss = (loss_left + loss_right)/2.0 + 4.0 * loss_gaze
            total_loss /= batch_size

            loss_components = {"gaze_final": loss_gaze, "loss_left": loss_left, "loss_right": loss_right}

            return total_loss, loss_components
            # return  loss_left + loss_val_right + loss_gaze

        return deep_pictorial_loss_fn

    def create_optimizer(self):
        return super().create_optimizer()

    def create_summary_writer(self):
        return super().create_summary_writer()

    def train_val_test(self):
                # To meassure average epoch processing time
        epoch_time = AverageMeter()
        # Time of the beginning of an epoch
        start_time = time.time()  

        for curr_epoch in range(1, self.config["epochs"] + 1): 
            # Update current epoch count
            self.curr_epoch = curr_epoch

            # Log the estimated time for finishing the experiment
            self._log_estimated_epoch_time(epoch_time=epoch_time, lr=self.optimizer.param_groups[0]["lr"])

            # Train for one epoch
            train_loss, train_metric, train_loss_components = self.train_epoch()

            # Validate model on the validation set
            val_loss, val_metric, val_loss_components = self.validate_epoch()

            # Adjust the learning rate eafter each epoch, according to the lr scheduler
            self.lr_scheduler.step(val_loss)

            if self.summary_writer is not None:
                # Log metrics to the tensorboard
                self.summary_writer.add_scalars(f"Total loss",
                                                {"train": train_loss, "val": val_loss},
                                                self.curr_epoch)
                self.summary_writer.add_scalars(f"Metric",
                                                {"train": train_metric, "val": val_metric},
                                                self.curr_epoch)
                self.summary_writer.add_scalar(f"Learning rate", self.optimizer.param_groups[0]["lr"], self.curr_epoch)
                if train_loss_components is not None and val_loss_components is not None:
                    self.summary_writer.add_scalars(f"Gaze direction loss",
                                {"train": train_loss_components["gaze_final"], "val": val_loss_components["gaze_final"]},
                                self.curr_epoch)
                    for i, zippp in enumerate(zip(train_loss_components["loss_left"], val_loss_components["loss_left"], train_loss_components["loss_right"], val_loss_components["loss_right"])):
                        tr_int_loss_left, val_int_loss_left, tr_int_loss_right, val_int_loss_right = zippp
                        self.summary_writer.add_scalars(f"[{i}]Intermediate loss",
                                                    {"train_left": tr_int_loss_left, "val_left": val_int_loss_left,
                                                        "train_right": tr_int_loss_right, "val_right": val_int_loss_right},
                                                    self.curr_epoch)

            # Calculate epoch time, and restart timer
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

            # Save a checkpoint of the model after each epoch
            self.save_checkpoint()

            # Check if the training should stop due to early stopping
            if (self.curr_epoch - self.best_epoch) == self.config["early_stop"]: 
                self.print_to_log("EARLY STOPPING \n")
                break

        # Log the training report 
        self.training_end_report()

        # TODO Think later do we want to run it on test set here or not...
        # # Load the model with the best validation performance 
        # self.load_best_checkpoint()

        # Close the file for logging
        self.fh.close()

        # # After training has finished, evaluate the model on the train/val/test and log the results.
        # train_loss, train_metric = self.run_one_epoch(which="train", update_weights=False)
        # val_loss, val_metric = self.run_one_epoch(which="val", update_weights=False)

        return

    def train_epoch(self):
        # Train for one epoch
        train_loss, train_metric, loss_components = self.run_one_epoch(which="train", update_weights=True)

        # Append the loss and acc to the list (save one value for evey epoch)
        self.train_loss.append(train_loss)
        self.train_metric.append(train_metric)

        return train_loss, train_metric, loss_components

    def validate_epoch(self):
        # Validate for one epoch
        val_loss, val_metric, loss_components = self.run_one_epoch(which="val", update_weights=False)

        # If the current validation loss is better than from all previous epochs
        if self.curr_epoch == 1:
            self.best_epoch = 1
        elif val_loss < np.min(self.val_loss):
            self.best_epoch = self.curr_epoch

        # Append the loss and acc to the list (save one value for evey epoch)
        self.val_loss.append(val_loss)
        self.val_metric.append(val_metric)

        return val_loss, val_metric, loss_components
    def run_one_epoch(self, which, update_weights):

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

    


    def calculate_metric(self, pred, gt):
        return super().calculate_metric(pred,gt)



