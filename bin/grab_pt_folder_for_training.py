#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Takes a list of pt names and preps their files to go into the lesion training folder
"""

import os
import shutil
from glob import glob

import pandas as pd
import numpy as np


master_folder = '/Users/manusdonahue/Documents/Sky/segmentations_sci/pt_data/'
to_folder = '/Users/manusdonahue/Documents/Sky/lesion_training_data/'

mr_ids = [
          'SCD_TRANSP_P002_02'
          ]


#####

folders_in_target = np.array(glob(os.path.join(to_folder, '*/'))) # list of all possible subdirectories
mr_ids_in_target = [os.path.basename(os.path.normpath(i)) for i in folders_in_target]

dups = [i for i in mr_ids if i in mr_ids_in_target]
if dups:
    raise Exception(f'{dups} are already present in target folder')


for i, pt_id in enumerate(mr_ids):
    
    
    print(f'{i+1} of {len(mr_ids)}: {pt_id}')
    
    target_folder = os.path.join(to_folder, pt_id)
    os.mkdir(target_folder)
    
    pt_bin_folder = os.path.join(master_folder, pt_id, 'bin')
    pt_proc_folder = os.path.join(master_folder, pt_id, 'processed')
    
    source_flair = os.path.join(pt_bin_folder, 'axFLAIR_raw.nii.gz')
    source_flair_cor = os.path.join(pt_bin_folder, 'corFLAIR_raw.nii.gz')
    source_t1 = os.path.join(pt_bin_folder, 'axT1_raw.nii.gz')
    #source_mask = os.path.join(pt_proc_folder, 'axFLAIR_mask.nii.gz')
    
    target_flair = os.path.join(target_folder, 'axFLAIR.nii.gz')
    target_flair_cor = os.path.join(target_folder, 'corFLAIR.nii.gz')
    target_t1 = os.path.join(target_folder, 'axT1.nii.gz')
    #target_mask = os.path.join(target_folder, 'axFLAIR_mask.nii.gz')
    
    sources = [source_flair, source_flair_cor, source_t1]
    targets = [target_flair, target_flair_cor, target_t1]
    
    for s,t in zip(sources,targets):
        shutil.copy(s,t)
