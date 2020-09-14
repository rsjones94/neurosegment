#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 17:32:15 2020

geometry based sieving
"""

import os
import pickle

from skimage import measure
import pandas as pd
import numpy as np
import nibabel as nib
from skimage import measure
from sklearn import neighbors


# a dictionary of the properties to generate, where the keys are the properties and the entries are the min-maxes to normalize at
PROPERTIES = {
                'area': (0,1e4),
                'extent': (0,1e3),
                'filled_area': (0,1e4),
                'inertia_tensor': (0,1e1),
                'major_axis_length': (0,5e1),
                'minor_axis_length': (0,5e1)
              }

properties = ['area', 'extent', 'filled_area', 'inertia_tensor', 'major_axis_length', 'minor_axis_length']

def read_nifti(img_path):
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
    
    return img


def generate_properties(im, props=PROPERTIES):
    """
    Generates geometric properties for shapes in the binary input image
    

    Parameters
    ----------
    im : TYPE
        DESCRIPTION.
    props : TYPE, optional
        DESCRIPTION. The default is ['area', 'extent', 'filled_area', 'inertia_tensor', 'major_axis_length', 'minor_axis_length'].

    Returns
    -------
    X_train : TYPE
        DESCRIPTION.

    """
    labeled = measure.label(im)
    X_train = pd.DataFrame(measure.regionprops_table(labeled, properties=props))
    return X_train


def train_and_save(training_data, outloc):
    """
    Trains a LOF algorithm for the purposes of novelty detection and pickles it
    

    Parameters
    ----------
    training_data : TYPE
        a pandas DataFrame of the training data.

    Returns
    -------
    the model.

    """

    lof = neighbors.LocalOutlierFactor(novelty=True)
    lof.fit(training_data)
    
    pickle.dump(lof, open(outloc, "wb" ))
    
    return lof
    

def load_default_model():
    script_folder = os.path.dirname(os.path.realpath(__file__))
    repo_folder = os.path.dirname(script_folder)
    model_loc = os.path.join(repo_folder, 'bin', 'gbs_models', 'gbs_default.pkl')
    
    lof = pickle.load(open(model_loc, 'rb'))
    
    return lof


def sieve_image(im, model=None, props=None):
    
    if model is None:
        model = load_default_model()
    if props is None:
        props=PROPERTIES
    
    labeled = measure.label(im)
    observations = pd.DataFrame(measure.regionprops_table(labeled, properties=props))
    labels_only = pd.DataFrame(measure.regionprops_table(labeled, properties=['label']))
    
    predictions = model.predict(observations)
    labels_only['prediction'] = predictions
    
    to_zero = [row['label'] for i, row in labels_only.iterrows() if row['prediction'] == -1]
    
    mask = np.isin(labeled, to_zero)
    
    new_im = im.copy()
    new_im[mask] = 0
    
    return new_im
    
    