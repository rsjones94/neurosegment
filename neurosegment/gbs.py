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
"""
PROPERTIES = {
                'area': (0,1e3),
                'extent': (0,1),
                'filled_area': (0,1e3),
                'inertia_tensor': (-100,100),
                'major_axis_length': (0,300),
                'minor_axis_length': (0,300)
              }
"""
PROPERTIES = ['area', 'extent', 'filled_area', 'inertia_tensor', 'major_axis_length', 'minor_axis_length']

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


def standardize_data(data, params):
    
    standard_data = data.copy()
    
    for col, mean, stddev in zip(data.columns, params[0], params[1]):
        standard_data[col] = (standard_data[col] - mean) / stddev
        
    return standard_data


def train_and_save(training_data, outloc):
    """
    Trains a LOF algorithm for the purposes of novelty detection and pickles it
    Standardizes the data first (transforms each column by subtracting the mean
    and then dividing by the stddev)
    

    Parameters
    ----------
    training_data : TYPE
        a pandas DataFrame of the training data.
    out_loc : TYPE
        name of the pickled object to save, which is a tuple with length 2, where
        the first entry is the model. The second is a list of lists, where the first
        list is the list of means used to transform the data and the second is the list
        of the stddevs used to transform the data

    Returns
    -------
     a tuple with length 2, where
        the first entry is the model. The second is a list of lists, where the first
        list is the list of means used to transform the data and the second is the list
        of the stddevs used to transform the data

    """
    standard_data = training_data.copy()
    
    means = []
    stddevs = []
    for col in training_data.columns:
        mean = training_data[col].mean()
        stddev = training_data[col].std()
        
        means.append(mean)
        stddevs.append(stddev)
        
    standard_data = standardize_data(training_data, (means, stddevs))
        

    lof = neighbors.LocalOutlierFactor(novelty=True)
    lof.fit(standard_data)
    
    out_obj = (lof, (means, stddevs))
    pickle.dump(out_obj, open(outloc, "wb" ))
    
    return out_obj
    

def load_default_model():
    script_folder = os.path.dirname(os.path.realpath(__file__))
    repo_folder = os.path.dirname(script_folder)
    model_loc = os.path.join(repo_folder, 'bin', 'gbs_models', 'gbs_default.pkl')
    
    lof, params = pickle.load(open(model_loc, 'rb'))
    
    return lof, params


def sieve_image(im, model_and_params=None, props=None):
    
    if model_and_params is None:
        model_and_params = load_default_model()
    if props is None:
        props=PROPERTIES
    
    model = model_and_params[0]
    params = model_and_params[1]
    
    labeled = measure.label(im)
    observations = pd.DataFrame(measure.regionprops_table(labeled, properties=props))
    labels_only = pd.DataFrame(measure.regionprops_table(labeled, properties=['label']))
    
    standard_observations = standardize_data(observations, params)
    
    predictions = model.predict(standard_observations)
    labels_only['prediction'] = predictions
    
    to_zero = [row['label'] for i, row in labels_only.iterrows() if row['prediction'] == -1]
    
    mask = np.isin(labeled, to_zero)
    
    new_im = im.copy()
    new_im[mask] = 0
    
    return new_im
    
    