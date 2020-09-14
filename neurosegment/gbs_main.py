#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 17:31:19 2020

@author: skyjones
"""

import os

import numpy as np
import pandas as pd
import scipy

import gbs


# for generating a default model

master_csv = '/Users/manusdonahue/Documents/Sky/segmentations_sci/pt_data/move_and_prepare_tabular_24-07-20-09_53.csv'
to_train_col = 'training'
pt_id_col = 'id'
master_folder = '/Users/manusdonahue/Documents/Sky/segmentations_sci/pt_data/'
out_model = 'gbs_default.pkl'

##########

script_folder = os.path.dirname(os.path.realpath(__file__))
repo_folder = os.path.dirname(script_folder)
out_model = os.path.join(repo_folder, 'bin', 'default_model', out_model)

df = pd.read_csv(master_csv)

training_data = pd.DataFrame()
for pt, do_train in zip(df[pt_id_col], df[to_train_col]):
    if do_train != 1:
        continue
    print(f'Pulling data for {pt}')
    
    folder = os.path.join(master_folder, pt, 'processed')
    lesion_file = os.path.join(folder, 'axFLAIR_mask.nii.gz')
    lesion_im = gbs.read_nifti(lesion_file)
    
    lesion_info = gbs.generate_properties(lesion_im)
    training_data = training_data.append(lesion_info)
    

print('Saving model')
lof = gbs.train_and_save(training_data, out_model)
