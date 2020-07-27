#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for training BIANCA models

"""

import os

import pandas as pd


def generate_master(top_folder, master_name, training_subfolder,
                    training_names, in_csv, incl_col, pt_id_col):
    """
    Generates the master .txt file for input to BIANCA
    

    Parameters
    ----------
    top_folder : str
        path to the top level folder containing training data, where each subfolder is the pt ID.
    master_name : str
        out .txt file
    training_subfolder : str
        additional path extension to the training folder from the top_folder. the same names should be used for each training_subfolder's files
    training_names : list of str
        stems for the training files. rows will be built in order of the names specified
    in_csv : str
        path to csv that contains at least two columns relating pt ID to whether the pt has training masks.
    incl_col : str
        the name of the column indicating if the pt should be included in the master file.
    pt_id_col : str
        name of the column with the pt ID (matching the subfolder names in top_folder)

    Returns
    -------
    None

    """
    
    df = pd.read_csv(in_csv)
    df = df[df[incl_col] == 1]
    
    message_file = open(master_name, 'w')

    
    n_files = None
    for index, row in df.iterrows():
        
        pt_id = row[pt_id_col]
        target = os.path.join(top_folder, pt_id, training_subfolder)
        files = [os.path.join(target,f) for f in os.listdir(target) if os.path.isfile(os.path.join(target, f))]
        files = sorted(files)
        if len(files) != n_files and n_files != None:
            raise Exception(f'Number of files does not match. Expected {n_files} but got {len(files)} in {target}')
        n_files = len(files)
        for f in files:
            message_file.write(f'{f} ')
        message_file.write('\n')
        
    message_file.close()


def construct_bianca_cmd(master_name, subject_index, skullstrip_col, mask_col, out_name):
    """
    Constructs a string that can be passed to the OS to execute BIANCA
    

    Parameters
    ----------
    master_name : str
        path to the master txt file.
    subject_index : int
        1-indexed index of the row of the subject to segment.
    skullstrip_col : int
        1-indexed index of the column that contains a skullstripping mask (usually use the FLAIR).
    mask_col : int
        1-indexed index of the column that contains the brain lesion mask.
    out_name : str
        name of the trained BIANCA model to write.

    Returns
    -------
    executable string.

    """
    
    bianca = f'fsl bianca --singlefile={master_name} --querysubjectnum={subject_index} --brainmaskfeaturenum={skullstrip_col} --labelfeaturenum={mask_col} -o {out_name}'

    return bianca
    
