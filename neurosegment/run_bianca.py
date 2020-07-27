#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates and validates with repeated leave-one-out validation
"""
import os

import bianca_helpers as bh

tf = '/Users/manusdonahue/Documents/Sky/segmentations_sci/pt_data/'
mstn = '/Users/manusdonahue/Documents/Sky/segmentations_sci/bianca/master.txt'
tsf = 'processed'
trn = ['axFLAIR.nii.gz', 'axT1.nii.gz', 'axFLAIR_mask.nii.gz']
icsv = '/Users/manusdonahue/Documents/Sky/segmentations_sci/pt_data/move_and_prepare_tabular_24-07-20-09_53.csv'
tc = 'training'
pidc = 'id'


col_n = bh.generate_master(tf, mstn, tsf, trn, icsv, tc, pidc)
excl = 1
cmd = bh.construct_bianca_cmd(mstn, excl, 1, 3, '/Users/manusdonahue/Documents/Sky/segmentations_sci/bianca/test_model/tester')

os.system(cmd)