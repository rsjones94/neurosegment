#!/usr/bin/env python

"""
Executable that reads in a list of patient IDs, finds specified imagery,
converts them to NiFTI format, skullstrips and coregisters them (if desired).

Also does FAST segmentation

Takes the following arguments in the shell
    -i : a comma-delimited csv containing columns of pt IDs
    -f : the folder that contains subfolder of pt data. can be nested
    -t : the folder to write each pts data to
    -o : an integer bool (0 or 1) indicating whether to overwrite a pts data
        if they already have a subfolder in the target folder

The script will output a txt file to the target folder indicating what pts, if
any, could not be located, and if any FLAIR or T1 data could not be found.

Example use:
    
    /Users/manusdonahue/Documents/Sky/repositories/neurosegment/bin/move_and_prepare -i /Users/manusdonahue/Documents/Sky/stroke_segmentations_playground/pts_of_interest.csv -t /Users/manusdonahue/Documents/Sky/stroke_segmentations_playground/pt_data -f /Volumes/DonahueDataDrive/Data_sort/SCD_Grouped -o 0
"""

import os
import sys
import getopt
import glob
import shutil
from datetime import datetime
from contextlib import contextmanager
from time import time, sleep

import pandas as pd
import numpy as np

np.random.seed(0)

# I am a liar this script is now accessed directly rather than as a bash command

overwrite = 0

infile = '/Users/manusdonahue/Documents/Sky/nigeria_mra/orig_report_labels.csv'

targetfolder = '/Users/manusdonahue/Documents/Sky/nigeria_mra/data/'

filefolder = '/Volumes/DonahueDataDrive/Data_sort/SCD_Grouped/'

skullstrip_f_val = 0.15

n_healthy = 100


##### the following variables generally just need to be set once

# column names in the csv that contain pt IDs of interest
#pt_id_cols = ['MRI 1 - MR ID', 'MRI 2 - MR ID', 'MRI 3 - MR ID']
pt_id_cols = ['MRI 1 - MR ID']
pt_id_cols_alt = ['Alternate MR ID 1']
rect_name = ['MR 1 ID Rectified']

# dcm2nii is an executable packaged with MRIcron that ca be ued to turn par-recs into NiFTIs
path_to_dcm2nii = '/Users/manusdonahue/Documents/Sky/mricron/dcm2nii64'

mni_standard = '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'

# relates a unique sequence in a filename that can be used to identify a file
# as a certain type of image to the basename for the ouput, whether and what to
# register them to, and whether to skullstrip them
# signatures work such that if the key is found in the filename and the excl
# strings are NOT found in the filename, then that file is IDd as that signature

# note that duplicate patients who were given an alternate study ID need to be manually removed


signature_relationships = {('MRA_COW','TOF_HEAD'):
                              {'basename': 'headMRA', 'register': 'master', 'skullstrip': 'no', 'excl':['MIP'], 'optional':False},
                           ('MIP*MRA_COW','MIP*TOF_HEAD'):
                              {'basename': 'headMRA_mip', 'register': 'no', 'skullstrip': 'no', 'excl':[], 'optional':True}
                          }

"""
signature_relationships = {('MRA_COW','TOF_HEAD'):
                              {'basename': 'headMRA', 'register': 'master', 'skullstrip': 'no', 'excl':['WIP_MIP', 'MIP_WIP'], 'optional':True},
                           ('TOF_NECK',):
                              {'basename': 'neckMRA', 'register': 'no', 'skullstrip': 'no', 'excl':['WIP_MIP', 'MIP_WIP', 'VWIP'], 'optional':True}
                          }
 
signature_relationships = {('MRA_COW',):
                              {'basename': 'headMRA', 'register': 'master', 'skullstrip': 'no', 'excl':['WIP_MIP', 'MIP_WIP']},
                          ('lica',):
                              {'basename': 'pc_lica', 'register': 'no', 'skullstrip': 'no', 'excl':[]},
                          ('rica',):
                              {'basename': 'pc_rica', 'register': 'no', 'skullstrip': 'no', 'excl':[]},
                          ('lvert',):
                              {'basename': 'pc_lvert', 'register': 'no', 'skullstrip': 'no', 'excl':[]},
                          ('rvert',):
                              {'basename': 'pc_rvert', 'register': 'no', 'skullstrip': 'no', 'excl':[]},
                          }
"""

def any_in_str(s, l):
    """
    Returns whether any of a list of substrings is in a string
    

    Parameters
    ----------
    s : str
        string to look for substrings in.
    l : list of str
        substrings to check for in s.

    Returns
    -------
    bool

    """
    
    return any([substr in s for substr in l])

start = time()

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def get_terminal(path):
    """
    Takes a filepath or directory tree and returns the last file or directory
    

    Parameters
    ----------
    path : path
        path in question.

    Returns
    -------
    str of only the final file or directory.

    """
    
    return os.path.basename(os.path.normpath(path))

successful = 0


# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d-%m-%y-%H+%M")
message_file_name = os.path.join(targetfolder, f'move_and_prepare_messages_{dt_string}.txt')
df_file_name = os.path.join(targetfolder, f'move_and_prepare_tabular_{dt_string}.csv')
trimmed_file_name = os.path.join(targetfolder, f'pt_data.csv')
message_file = open(message_file_name, 'w')
message_file.write('Status messages for move_and_prepare\n\nSignatures')
for key, val in signature_relationships.items():
    message_file.write(f'\n{key}\n\t{str(val)}')
message_file.write('\n\n\n')

#extract pt IDs
raw_data = pd.read_csv(infile)

for i,val in enumerate(pt_id_cols):
    mrs = raw_data[pt_id_cols[i]]
    mrs_alt = raw_data[pt_id_cols_alt[i]]
    raw_data[rect_name[i]] = mrs_alt.combine_first(mrs)
    

#not_excluded = raw_data['Exclude from Analysis  (choice=exclude)'] == 'Unchecked'
not_inadequate = raw_data['Result of MRA Head 1'] != 'Technically inadequate'
is_done = raw_data['Result of MRA Head 1'] != 'Not done'
not_post_transf = raw_data['Is this patient post-transplant at initial visit?'] != 'Yes'
normal_or_scd = [any([i,j]) for i,j in zip(raw_data['Hemoglobin genotype'] == 'Normal (AA)', raw_data['Hemoglobin genotype'] == 'SS')]

#keep = [all(i) for i in zip(not_excluded, not_inadequate, is_done, not_post_transf, normal_or_scd)]
keep = [all(i) for i in zip(not_inadequate, is_done, not_post_transf, normal_or_scd)] 
pt_data = raw_data[keep]

has_stenosis = pt_data[pt_data['Is there intracranial stenosis (>50%)?'] == 'Yes']
is_healthy = pt_data[pt_data['Result of MRA Head 1'] == 'Normal']
n_stenosis = len(has_stenosis)

add_healthy = is_healthy.sample(n_healthy)

pt_data = has_stenosis.append(add_healthy)

# FOR TESTING
# pt_data = pt_data.iloc[18:20]

n_unique_pts = len(pt_data)

print(f'We have {n_unique_pts} patients. {len(has_stenosis)} stenosis, {len(add_healthy)} healthy')
sleep(2)

pt_ids = []
for col in rect_name:
    of_interest = list(pt_data[col])
    pt_ids.extend(of_interest)

pt_ids = [x for x in pt_ids if str(x) != 'nan']

# create a nested dict giving the the status of each pt id (found their file, found specific scans)
inner_dict = {'found_pt':0}
for key in signature_relationships:
    inner_dict[key] = (0,0)
inner_dict['successful'] = 0
pt_status = {pt:inner_dict.copy() for pt in pt_ids}

# start processing
all_subdirectories = [x[0] for x in os.walk(filefolder)] # list of all possible subdirectories

for i, pt in enumerate(pt_ids):
        print(f'\nOn patient {pt} ({i+1} of {len(pt_ids)})\n')
        candidate_folders = [sub for sub in all_subdirectories if get_terminal(sub) == pt] # check if last subfolder is pt name
        n_cands = len(candidate_folders)
        pt_status[pt]['found_pt'] = n_cands
        if n_cands == 1:
            data_folder = candidate_folders[0]
        else:
            print(f'------ pt {pt} has {n_cands} candidate folders. skipping ------')
            continue
            
        master_output_folder = os.path.join(targetfolder, pt)
        if os.path.exists(master_output_folder):
            if overwrite:
                shutil.rmtree(master_output_folder)
            else:
                print(f'--- pt {pt} exists in target folder and overwrite is disabled. skipping ---')
                continue
            
        
        has_required_files = True
        
        bin_folder = os.path.join(master_output_folder, 'bin') # bin for working with data
        processed_folder = os.path.join(master_output_folder, 'processed') # where we'll write the final data to
        acquired_folder = os.path.join(data_folder, 'Acquired') # where we're looking to pull data from
        
        sig_tracker = {} # to store filepaths to files
        optional_and_missing = []
        for signature, subdict in signature_relationships.items():
            
            candidate_pars = []
            candidate_recs = []
            # note that the signature matching includes the full path. probably not a great idea
            for subsig in signature:
                
                potential_pars = glob.glob(os.path.join(acquired_folder, f'*{subsig}*.PAR'))
                potential_recs = glob.glob(os.path.join(acquired_folder, f'*{subsig}*.REC'))
                
                potential_pars = [f for f in potential_pars if not any_in_str(f, subdict['excl'])]
                potential_recs = [f for f in potential_recs if not any_in_str(f, subdict['excl'])]
                
                candidate_pars.extend(potential_pars)
                candidate_recs.extend(potential_recs)
            
            n_cand_files = (len(candidate_pars), len(candidate_recs))
            pt_status[pt][signature] = n_cand_files
            if all([i >= 1 for i in n_cand_files]):
                
                sig_tracker[signature] = {'original_par': candidate_pars[-1]}
                sig_tracker[signature]['original_rec'] = candidate_recs[-1]
                
                moved_par = os.path.join(bin_folder, get_terminal(candidate_pars[-1]))
                moved_rec = os.path.join(bin_folder, get_terminal(candidate_recs[-1]))
                
                sig_tracker[signature]['moved_par'] = moved_par
                sig_tracker[signature]['moved_rec'] = moved_rec
                
                if any(i != 1 for i in n_cand_files):
                    print(f'warning: pt {pt} returned {n_cand_files} for {signature}. using last option')
            else:
                if subdict['optional']:
                    print(f'pt {pt} has {n_cand_files} candidate par/recs for {signature}, but this an optional signature')
                    optional_and_missing.append(signature)
                else:
                    print(f'pt {pt} has {n_cand_files} candidate par/recs for {signature}. will be skipped')
                    has_required_files = False
            
        if not has_required_files: # if we don't have all the files specified, just move on
            continue    
        
        os.mkdir(master_output_folder)
        os.mkdir(bin_folder)
        os.mkdir(processed_folder)
        
        try:
            for signature, subdict in signature_relationships.items():
                if signature in optional_and_missing:
                    continue
                # move the file, convert to NiFTI and rename
                shutil.copyfile(sig_tracker[signature]['original_par'], sig_tracker[signature]['moved_par'])
                shutil.copyfile(sig_tracker[signature]['original_rec'], sig_tracker[signature]['moved_rec'])
                
                moved_par_without_ext = sig_tracker[signature]['moved_par'][:-4]
                conversion_command = f'{path_to_dcm2nii} -o {bin_folder} -a n -i n -d n -p n -e n -f y -v n {sig_tracker[signature]["moved_par"]}'
                
                with suppress_stdout():
                    os.system(conversion_command)
                
                sig_tracker[signature]['raw_nifti'] = os.path.join(bin_folder, f'{subdict["basename"]}_raw.nii.gz')
                os.rename(f'{moved_par_without_ext}.nii.gz', sig_tracker[signature]['raw_nifti'])
        except:
            print(f'\n!!!!!!!!!! warning: encountered unexpected error while copying and converting images for pt {pt}. folder will be deleted !!!!!!!!!!\n')
            shutil.rmtree(master_output_folder)
            continue
        
        # skullstripping
        for signature, subdict in signature_relationships.items():
            if signature in optional_and_missing:
                continue
            if subdict['skullstrip'] == 'yes':
                sig_tracker[signature]['skullstripped_nifti'] = os.path.join(bin_folder, f'{subdict["basename"]}_stripped.nii.gz')
                
                stripping_command = f"bet {sig_tracker[signature]['raw_nifti']} {sig_tracker[signature]['skullstripped_nifti']} -f {skullstrip_f_val}"
                os.system(stripping_command)
                
            else:
                sig_tracker[signature]['skullstripped_nifti'] = sig_tracker[signature]['raw_nifti']
            
        # registration
        for signature, subdict in signature_relationships.items():
            if signature in optional_and_missing:
                continue
            if subdict['register'] == 'master':
                master_ref = sig_tracker[signature]['skullstripped_nifti']
                
                omat_path = os.path.join(processed_folder, 'master2mni.mat')
                mni_path = os.path.join(bin_folder, f'{subdict["basename"]}_mni.nii.gz')
                omat_cmd = f'flirt -in {master_ref} -ref {mni_standard} -out {mni_path} -omat omat_path'
                os.system(omat_cmd)
                
        for signature, subdict in signature_relationships.items():
            if signature in optional_and_missing:
                continue
            if subdict['register'] not in ('master', 'no'):
                sig_tracker[signature]['registered_nifti'] = os.path.join(bin_folder, f'{subdict["basename"]}_registered.nii.gz')
                register_command = f"flirt -in {sig_tracker[signature]['skullstripped_nifti']} -ref {master_ref} -out {sig_tracker[signature]['registered_nifti']}"
                os.system(register_command)
            else:
                sig_tracker[signature]['registered_nifti'] = sig_tracker[signature]['skullstripped_nifti']
                
        # move files to their final home :)
        for signature, subdict in signature_relationships.items():
            if signature in optional_and_missing:
                continue
            sig_tracker[signature]['final_nifti'] = os.path.join(master_output_folder, f'{subdict["basename"]}.nii.gz')
            shutil.copyfile(sig_tracker[signature]['registered_nifti'], sig_tracker[signature]['final_nifti'])
                
        # delete the subfolders
            
        folder_glob = np.array(glob.glob(os.path.join(master_output_folder, '*/'))) # list of all possible subdirectories
        for f in folder_glob:
            shutil.rmtree(f)
            
        # if the master folder is empty, that means that all files were optionally, but none were found. I'd call that a failure and delete the folder
        # but otherwise it's fine
        file_glob = np.array(glob.glob(os.path.join(master_output_folder, '*'))) # list of all files
        if len(file_glob) == 0:
            print(f'No files transferred to {master_output_folder}: deleting folder and marking as failure')
            pt_status[pt]['successful'] = 0
            successful += 0
            shutil.rmtree(master_output_folder)
        else: 
            pt_status[pt]['successful'] = 1
            successful += 1
        


# write status log
            
end = time()

runtime = end-start
runtime_minutes_pretty = round(runtime/60, 2)
        
message_file.write(f'Successfully preprocessed {successful} of {len(pt_ids)} scans from {n_unique_pts} unique patients. Running time: {runtime_minutes_pretty} minutes\n\n\n')
df = pd.DataFrame()
for key, val in pt_status.items():
    message_file.write(f'Patient {key}\n\t{str(val)}\n\n')
    appender = pd.Series(val, name=key)
    df = df.append(appender)
    

message_file.close()
df.to_csv(df_file_name)


only_success = df[df['successful']==1]

keep_in_trim = [True if i in only_success.index else False for i in pt_data[rect_name[0]]]
trim_pt_data = pt_data[keep_in_trim]
trim_pt_data.to_csv(trimmed_file_name)