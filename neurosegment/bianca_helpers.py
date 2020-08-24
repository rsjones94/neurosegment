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
    
    assert len(df) != 0
    
    message_file = open(master_name, 'w')

    i = 0
    for index, row in df.iterrows():
        
        pt_id = row[pt_id_col]
        target = os.path.join(top_folder, pt_id, training_subfolder)
        files = [os.path.join(target,f) for f in training_names]
        if not all([os.path.exists(f) for f in files]):
            raise Exception(f'Folder {target} does not contain all specified files')
        for f in files:
            message_file.write(f'{f} ')
        if i != len(df)-1:
            message_file.write('\n')
        i += 1
        
    message_file.close()


def construct_bianca_cmd(master_name, subject_index, skullstrip_col, mask_col, out_name, run_cmd=True):
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
    
    bianca = f'bianca --singlefile={master_name} --querysubjectnum={subject_index} --trainingnums=all --brainmaskfeaturenum={skullstrip_col} --selectpts=surround --trainingpts=equalpoints --labelfeaturenum={mask_col} --saveclassifierdata={out_name}_classifer -o {out_name}'
    if run_cmd:
        os.system(bianca)
    return bianca
    
def evaluate_bianca_performance(bianca_mask, thresh, manual_mask, run_cmd=True):
    """
    Evaluates the quality of a BIANCA lesion segmentation
    

    Parameters
    ----------
    bianca_mask : str
        the name of the of fuzzy BIANCA lesion mask to be evaluated.
    thresh : float
        DESCRIPTION.
    manual_mask : str
        name of the manual binary lesion mask.
    run_cmd : bool
        if True, execute the generated command.

    Returns
    -------
    evaluation_cmd : TYPE
        DESCRIPTION.

    """
    
    evaluation_cmd = f'bianca_overlap_measures {bianca_mask} {thresh} {manual_mask} 1'
    # bianca_overlap_measures <lesionmask> <threshold> <manualmask> <saveoutput>
    if run_cmd:
        os.system(evaluation_cmd)
    return evaluation_cmd