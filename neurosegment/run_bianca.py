#!/usr/bin/env python3
# -*- coding: utraining_folder-8 -*-
"""
Generates BIANCA models and validates with repeated leave-one-out validation
"""
import os
import time

import pandas as pd

import bianca_helpers as bh

training_folder = '/Users/manusdonahue/Documents/Sky/segmentations_sci/pt_data/'
master_file_path = '/Users/manusdonahue/Documents/Sky/segmentations_sci/bianca/master.txt'
training_subfolder = 'processed'
training_stems = ['axFLAIR.nii.gz', 'axT1.nii.gz', 'axFLAIR_mask.nii.gz']
input_csv = '/Users/manusdonahue/Documents/Sky/segmentations_sci/pt_data/move_and_prepare_tabular_24-07-20-09_53.csv'
training_boolean_column_header = 'training'
pt_id_col_header = 'id'

brainmask_col = 1
trainingmask_col = 3

thresh = 0.5

validation_folder = '/Users/manusdonahue/Documents/Sky/segmentations_sci/bianca/validation/'
####


col_n = bh.generate_master(training_folder, master_file_path, training_subfolder,
                           training_stems, input_csv, training_boolean_column_header, pt_id_col_header)

master_file = open(master_file_path, 'r')
content = master_file.read()
content_list = content.split('\n')

report_name = os.path.join(validation_folder, 'BIANCA_report.csv')

labels = ['SI', 'FDR', 'FNR', 'FDR_clus', 'FNR_clus', 'MTA', 'DER', 'OER', 'BIANCA_vol', 'manual_vol']
report = pd.DataFrame(columns=labels)

start = time.time()

for i, row in enumerate(content_list):
    
    excl = i+1 # the row that will be left out
    
    elap = time.time() - start
    print(f'Generating BIANCA model (leave-one-out: {excl}). Elapsed time: {round(elap/60, 2)} minutes')
    
    target_folder = os.path.join(validation_folder, f'leave_out_{excl}')
    output_name = os.path.join(target_folder, 'raw_bianca')
    output_name_with_ext = output_name + '.nii.gz'
    os.mkdir(target_folder)
    
    bianca_cmd = bh.construct_bianca_cmd(master_file_path,
                                  excl,
                                  brainmask_col,
                                  trainingmask_col,
                                  output_name)
    
    files_on_row = row.split(' ')
    manual_mask = files_on_row[trainingmask_col-1]
    
    print(f'Evaluating model performance\n')
    
    perf = bh.evaluate_bianca_performance(output_name_with_ext, thresh,
                                          manual_mask, run_cmd=True)
    perf_name = f'Overlap_and_Volumes_raw_bianca_{thresh}.txt'
    full_perf = os.path.join(target_folder, perf_name)
    
    """
    Performance evaluation:
        
    Dice Similarity Index (SI): calculated as 2*(voxels in the intersection of manual and BIANCA masks)/(manual mask lesion voxels + BIANCA lesion voxels)
    Voxel-level false discovery rate (FDR): number of voxels incorrectly labelled as lesion (false positives, FP) divided by the total number of voxels labelled as lesion by BIANCA (positive voxels)
    Voxel-level false negative ratio (FNR): number of voxels incorrectly labelled as non-lesion (false negatives, FN) divided by the total number of voxels labelled as lesion in the manual mask (true voxels)
    Cluster-level FDR: number of clusters incorrectly labelled as lesion (FP) divided by the total number of clusters found by BIANCA (positive clusters)
    Cluster-level FNR: number of clusters incorrectly labelled as non-lesion (FN) divided by the total number of lesions in the manual mask (true clusters)
    Mean Total Area (MTA): average number of voxels in the manual mask and BIANCA output (true voxels + positive voxels)/2
    Detection error rate (DER): sum of voxels belonging to FP or FN clusters, divided by MTA
    Outline error rate (OER): sum of voxels belonging to true positive clusters (WMH clusters detected by both manual and BIANCA segmentation), excluding the overlapping voxels, divided by MTA

    Volume of BIANCA segmentation (after applying the specified threshold)
    Volume of manual mask
    """
    # extract the performance evaluation metrics
    perf_data = open(full_perf, 'r')
    perf_text = perf_data.read()
    perf_cells = perf_text.split(' ')
    perf_cells = [float(i) for i in perf_cells]
    data_dict = {key:val for key,val in zip(labels, perf_cells)}
    data_sr = pd.Series(data_dict, name=excl)
    report = report.append(data_sr)
    
report.to_csv(report_name)

total_time = time.time() - start
print(f'Finished. Total running time: {round(total_time/60, 2)} minutes')
    
    