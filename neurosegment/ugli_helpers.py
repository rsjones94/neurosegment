#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:10:56 2020

@author: skyjones
"""

import os
import glob

import nibabel as nib
import numpy as np
from scipy import ndimage

def read_nifti_radiological(img_path):
    """
    Wrapper for nibabel to read in NIfTI scans.
    

    Parameters
    ----------
    img_path : string
        The path to the .nii scan
            
    Returns
    -------
    A numpy array representing the scan

    """
    raw = nib.load(img_path)
    img = raw.get_fdata()
    img = np.rot90(img, k=1)
    #img = ndimage.rotate(img.T, 180)
    # img = np.fliplr(img) # uncomment for neurological orientation, along with the writing method in uglu
    
    return img
    


def generate_bianca_master(parent_folder, flair, t1):
    # returns the name out the output master file
    
    master_name = os.path.join(parent_folder, 'bianca_master.txt')
    message_file = open(master_name, 'w')
    message_file.write(f'{flair} {t1}')
    message_file.close()
    return master_name


def execute_bianca(master, model, outname):
    """
    Generates a BIANCA probability map given flair, t1, a master file
    and a pretrained BIANCA model. Assumes the master file uses a column layout that mimics
    that used to train the model
    

    Parameters
    ----------
    master : pathlike
        path to BIANCA masterfil.
    model : pathlike
        path to pretrained model.
    outname : pathlike
        path of the output map

    Returns
    -------
    None

    """
    
    cmd = f'bianca --singlefile={master} --querysubjectnum={1} --brainmaskfeaturenum=1 --loadclassifierdata={model} -o {outname}'
    print(f'BIANCA execution: {cmd}')
    os.system(cmd)