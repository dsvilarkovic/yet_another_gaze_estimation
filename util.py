import os
import shutil
from datetime import datetime

import torch
import math


def createDirectory(dirPath, verbose = True):
    if not os.path.isdir(dirPath):
        os.makedirs(dirPath)
        if verbose:
            print("Folder not found!!!   " + dirPath + " created.")




def createCheckpointDir(outputFolderPath = '../Output/', debug_mode = False):
    
    ## Create output folder, if it does not exist
    createDirectory(outputFolderPath, verbose = False)
    
    ## Create folder to save current version, if it does not exist
    if debug_mode:
        outputCurrVerFolderPath = os.path.join( outputFolderPath, 'Checkpoint_ver_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_debug' )
    else:
        outputCurrVerFolderPath = os.path.join( outputFolderPath, 'Checkpoint_ver_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') )
        
        
    createDirectory(outputCurrVerFolderPath, verbose = False)
    print("Output will be saved to:  " + outputCurrVerFolderPath)
    
    return outputCurrVerFolderPath




def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
            

from zipfile import ZipFile

def copyToZip(sourceFolder  = './', destZipFileName = '../zipFile.zip', ignoreHidden = True):

    with ZipFile(destZipFileName, 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk(sourceFolder):
            
            if ignoreHidden:
                filenames = [f for f in filenames if not f[0] == '.']
                subfolders[:] = [d for d in subfolders if not d[0] == '.']
            
            for filename in filenames:
                #create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(filePath)

def mse_error(pred, gt):
    return torch.tensor(math.degrees(torch.mean(torch.abs(pred - gt))))

def angular_error(pred, gt, mean=True):

    v_pred = angle_to_unit_vectors(pred)
    v_gt = angle_to_unit_vectors(gt)

    return angular_error_from_vector(v_pred, v_gt, mean=mean)
    
def angle_to_unit_vectors(y):
    sin = torch.sin(y)
    cos = torch.cos(y)
    return torch.stack([
        cos[:, 0] * sin[:, 1],
        sin[:, 0],
        cos[:, 0] * cos[:, 1]],
        dim=1)

def angular_error_from_vector(v_pred, v_gt, mean=True):
    v_pred_norm = torch.norm(v_pred, p=2, dim=1)
    v_gt_norm = torch.norm(v_gt, p=2, dim=1)

    sim = torch.sum(v_pred * v_gt, dim=1) / (v_pred_norm * v_gt_norm + 1e-6)

    sim = torch.clamp(sim, -1.0, 1.0)

    ang = (180.0 / math.pi) * torch.acos(sim)

    if mean:
        return torch.mean(ang)
    else:
        return ang
