#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One off for preparing lesion segmentations for validation by a neuroradiologist
"""

import os
import shutil

import pandas as pd


master_folder = '/Users/manusdonahue/Documents/Sky/segmentations_sci/pt_data/'
to_folder = '/Users/manusdonahue/Documents/Sky/lesion_training_data/'

training_csv = '/Users/manusdonahue/Documents/Sky/segmentations_sci/pt_data/move_and_prepare_tabular_24-07-20-09_53.csv'


#####


table = pd.read_csv(training_csv)

is_one = table['training'] == 1
is_nought = table['training'] == 0
trutru = [any([i,j]) for i,j in zip(is_one, is_nought)]

table = table[trutru]


for i, row in table.iterrows():
    pt_id = row['id']
    
    print(f'{i+1} of {len(table)}: {pt_id}')
    
    target_folder = os.path.join(to_folder, pt_id)
    os.mkdir(target_folder)
    
    pt_bin_folder = os.path.join(master_folder, pt_id, 'bin')
    pt_proc_folder = os.path.join(master_folder, pt_id, 'processed')
    
    source_flair = os.path.join(pt_bin_folder, 'axFLAIR_raw.nii.gz')
    source_flair_cor = os.path.join(pt_bin_folder, 'corFLAIR_raw.nii.gz')
    source_t1 = os.path.join(pt_bin_folder, 'axT1_raw.nii.gz')
    source_mask = os.path.join(pt_proc_folder, 'axFLAIR_mask.nii.gz')
    
    target_flair = os.path.join(target_folder, 'axFLAIR.nii.gz')
    target_flair_cor = os.path.join(target_folder, 'corFLAIR.nii.gz')
    target_t1 = os.path.join(target_folder, 'axT1.nii.gz')
    target_mask = os.path.join(target_folder, 'axFLAIR_mask.nii.gz')
    
    sources = [source_flair, source_flair_cor, source_t1, source_mask]
    targets = [target_flair, target_flair_cor, target_t1, target_mask]
    
    for s,t in zip(sources,targets):
        shutil.copy(s,t)
