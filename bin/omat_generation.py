#!/usr/bin/env python

"""

One off for adding transformation matrices to a move_and_prepare output folder

"""

import os
import time

master_folder = '/Users/manusdonahue/Documents/Sky/segmentations_sci/pt_data'
processed_folder = 'processed'
bin_folder = 'bin'
master_scan = 'axFLAIR'
mni_standard = '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'


all_subs = [f.path for f in os.scandir(master_folder) if f.is_dir()]
n = len(all_subs)

start = time.time()

for i, sub in enumerate(all_subs):
    print(f'\n{sub}: {i+1} of {n}')
    the_scan = os.path.join(master_folder, sub, processed_folder, f'{master_scan}.nii.gz')
    omat_path = os.path.join(master_folder, sub, processed_folder, 'master2mni.mat')
    mni_path = os.path.join(master_folder, sub, bin_folder, f'{master_scan}_mni.nii.gz')
    omat_cmd = f'flirt -in {the_scan} -ref {mni_standard} -out {mni_path} -omat {omat_path}'
    #print(omat_cmd)
    os.system(omat_cmd)
    
    mid = time.time()
    elap = mid - start
    frac_done = (i+1)/n
    mult = 1 / frac_done
    total_run_time = elap*mult
    time_remaining = total_run_time-elap
    
    pretty_elap = round(elap/60,1)
    pretty_time_remaining = round(time_remaining/60,1)
    
    print(f'{pretty_elap} minutes elapsed')
    print(f'{pretty_time_remaining} minutes remaining (estimated)')