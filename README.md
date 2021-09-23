# Eye Gaze Estimation 

[Report link](https://gitlab.inf.ethz.ch/COURSE-MP2020/Cyclopes/-/blob/master/Machine_Perceptron_Report___Nikola__Nikhil__Dusan.pdf)
## Setup
For setting up all needed modules on Leonhard, run these commands:
```
module load eth_proxy
module load gcc/6.3.0
module load cuda/10.0.130
module load cudnn/7.6.4
module load nccl/2.4.8-1
module load openblas/0.2.19
module load jpeg/9b
module load libpng/1.6.27
```
For setting up an appropriate conda environment:

```
conda create -n machine_perception_env python=3.8
conda activate machine_perception_env
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install scipy pandas numpy scikit-image h5py tensorboard
pip install torchsummary opencv-python
```


## Training on the training and validation set + Inference on the test set
Our best model configuration can be found as a json file: **configs/best_model_assymetric_updated_densenet161_lr_0.0001_wd_0.005.json** <br/>

In that file, there is a **"code_dir"** which is the code absolute location where you want to save the whole codebase when submitting a job. If you want to avoid that, set  **"code_dir" : null**.   </br>
To run our best model, use the command: 
```
python running_with_config.py configs/best_model_assymetric_updated_densenet161_lr_0.0001_wd_0.005.json
```
To run it on Leonhard cluster, example would be:
```
bsub -W 120:00 -n 1 -R "rusage[mem=8092,ngpus_excl_p=1]" python running_with_config.py configs/best_model_assymetric_updated_densenet161_lr_0.0001_wd_0.005.json
```

This script both trains, predicts and saves files using the **save_output_dir** key, where you provide the absolute address of the folder where you want to save important files, and that key can be setup in the JSON config file (configs/best_model_assymetric_updated_densenet161_lr_0.0001_wd_0.005.json). Files saved under this folder include:

- training print log (**log_file.txt**)
- tensorboard logs (**tensorboard_logs/**)
- model checkpoints (**best_checkpoint.pth** or **checkpoint.pth**) 
- test submission zip files (**best_checkpoint_submit.txt.gz** or **last_checkpoint_submit.txt.gz**)
- codebase*

*if you run code by default running and no changes
If you leave **save_output_dir : null** as it is, your folder checkpoint will be located on **../Output** path (so, one level above your repo, make sure it exists).

***By executing the script above (running_with_config.py), you both train and inference your model on the test and in save_output_dir you have a .zip file used for submitting to a grader on http://machine-perception.ait.inf.ethz.ch/.***

## [Only] inference on the test set
***Warning: avoid using this, only used in case your batch submission finished before the epoch count training specified.***<br/>
To run only prediction on the test set with the checkpoint model you have (saved under the **save_output_dir**), you may use the **submit.py** script, where in a code, you have to define a location of the checkpoint folder where you saved your trained checkpoint *.pth model. For that, you need to change lines

```
checkpoint_file="Checkpoint_ver_2020-08-06_18-41-25.7728161596732085773"
chk_pt_dir = os.path.join(*["..", "Output", checkpoint_file])
```
to lines that suit your checkpoint folder. Also there is a config variable in the **submit.py** code which, needs to follow **configs/best_model_assymetric_updated_densenet161_lr_0.0001_wd_0.005.json** JSON file according to JSON standard in Python.

After that, run :

```
python submit.py 
```


Or if you are inferencing on Leonhard, it might look like this : 
```
bsub -W 00:30 -n 1 -R "rusage[mem=8092,ngpus_excl_p=1]" python submit.py
```


## Team Cyclopes:

- [Dusan Svilarkovic](https://gitlab.inf.ethz.ch/dsvilarko)
- [Nikola Popovic](https://gitlab.inf.ethz.ch/nipopovic)
- [Nikhil Prakash](https://gitlab.inf.ethz.ch/nprakash)
