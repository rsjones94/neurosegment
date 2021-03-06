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
from time import time

import numpy as np
import pandas as pd


# I am a liar this script is now accessed directly rather than as a bash command

overwrite = 0

infile = '/Users/manusdonahue/Documents/Sky/brain_lesion_masks/newest_additions/newest.csv'

targetfolder = '/Users/manusdonahue/Documents/Sky/brain_lesion_masks/newest_additions/'

filefolder = '/Volumes/DonahueDataDrive/Data_sort/SCD_Grouped/'

##### the following variables generally just need to be set once

# column names in the csv that contain pt IDs of interest
#pt_id_cols = ['MRI 1 - MR ID', 'MRI 2 - MR ID', 'MRI 3 - MR ID']
#pt_id_cols = ['mr1_mr_id_real', 'mr2_mr_id_real', 'mr3_mr_id_real']
pt_id_cols = ['mr1_mr_id_real']

# dcm2nii is an executable packaged with MRIcron that ca be ued to turn par-recs into NiFTIs
path_to_dcm2nii = '/Users/manusdonahue/Documents/Sky/mricron/dcm2nii64'

mni_standard = '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'

# relates a unique sequence in a filename that can be used to identify a file
# as a certain type of image to the basename for the ouput, whether and what to
# register them to, and whether to skullstrip them
# signatures work such that if the key is found in the filename and the excl
# strings are NOT found in the filename, then that file is IDd as that signature

signature_relationships = {('FLAIR_cor','FLAIR_COR'):
                              {'basename': 'corFLAIR', 'register': 'no', 'skullstrip': 'no', 'excl':['AX','ax','axial','AXIAL']},
                          ('FLAIR_AX', 'T2W_FLAIR'):
                              {'basename': 'axFLAIR', 'register': 'master', 'skullstrip': 'no', 'excl':['cor','COR','coronal','CORONAL']},
                          ('3DT1', 'T1W_3D'): 
                              {'basename': 'axT1', 'register': 'no', 'skullstrip': 'no', 'excl':['FLAIR']},
                          }

'''    
signature_relationships = {
                          ('3DT1', 'T1W_3D'): 
                              {'basename': 'axT1', 'register': 'master', 'skullstrip': 'no', 'excl':['FLAIR']},
                          }
'''

"""
signature_relationships = {
                          ('3DT1', 'T1W_3D'): 
                              {'basename': 'axT1', 'register': 'master', 'skullstrip': 'yes', 'excl':['FLAIR']},
                          }
"""
# list of dicts giving parameters for FAST
# inputs are image signatures
"""
fast_params = [
                {'inputs':['FLAIR_AX', '3DT1'], 'baseout':'fast_FLAIR+T1', 'n':4},
                {'inputs':['FLAIR_AX'], 'baseout':'fast_FLAIR', 'n':4},
                {'inputs':['3DT1'], 'baseout':'fast_T1', 'n':4}
              ]




fast_params = [
                {'inputs':[('3DT1', 'T1W_3D')], 'baseout':'fast_T1', 'n':3}
              ]
"""


fast_params = []

run_siena = False

skullstrip_f_val = 0.15 # variable for the BET skullstripping algorithm

#####

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


"""
bash_input = sys.argv[1:]
options, remainder = getopt.getopt(bash_input, "i:f:t:o:", ["infile=","filefolder=","targetfolder=","overwrite="])

for opt, arg in options:
    if opt in ('-o', '--overwrite'):
        overwrite = int(arg)
    elif opt in ('-i', '--infile'):
        infile = arg
    elif opt in ('-t', '--targetfolder'):
        targetfolder = arg
    elif opt in ('-f', '--filefolder'):
        filefolder = arg
        
        
try:
    assert overwrite in (0,1)
except AssertionError:
    raise AssertionError('overwrite argument must be 0 (for False) or 1 (for True)')
    
try:
    assert os.path.isdir(targetfolder)
except AssertionError:
    raise AssertionError('Target folder does not exist')
    
try:
    assert os.path.isdir(filefolder)
except AssertionError:
    raise AssertionError('File folder does not exist')
"""

# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d-%m-%y-%H+%M")
message_file_name = os.path.join(targetfolder, f'move_and_prepare_messages_{dt_string}.txt')
df_file_name = os.path.join(targetfolder, f'move_and_prepare_tabular_{dt_string}.csv')
message_file = open(message_file_name, 'w')
message_file.write('Status messages for move_and_prepare\n\nSignatures')
for key, val in signature_relationships.items():
    message_file.write(f'\n{key}\n\t{str(val)}')
message_file.write('\n\n\n')

#extract pt IDs
pt_data = pd.read_csv(infile)
n_unique_pts = len(pt_data)

pt_ids = []
for col in pt_id_cols:
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
        print(f'On patient {pt} ({i+1} of {len(pt_ids)})')
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
                
                sig_tracker[signature] = {'original_par': candidate_pars[0]}
                sig_tracker[signature]['original_rec'] = candidate_recs[0]
                
                moved_par = os.path.join(bin_folder, get_terminal(candidate_pars[0]))
                moved_rec = os.path.join(bin_folder, get_terminal(candidate_recs[0]))
                
                sig_tracker[signature]['moved_par'] = moved_par
                sig_tracker[signature]['moved_rec'] = moved_rec
                
                if any(i != 1 for i in n_cand_files):
                    print(f'warning: pt {pt} returned {n_cand_files} for {signature}. using first option')
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
                # move the file, convert to NiFTI and rename
                shutil.copyfile(sig_tracker[signature]['original_par'], sig_tracker[signature]['moved_par'])
                shutil.copyfile(sig_tracker[signature]['original_rec'], sig_tracker[signature]['moved_rec'])
                
                moved_par_without_ext = sig_tracker[signature]['moved_par'][:-4]
                conversion_command = f'{path_to_dcm2nii} -a n -i n -d n -p n -e n -f y -v n -o {bin_folder} {sig_tracker[signature]["moved_par"]}'
                
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
            if subdict['skullstrip'] == 'yes':
                sig_tracker[signature]['skullstripped_nifti'] = os.path.join(bin_folder, f'{subdict["basename"]}_stripped.nii.gz')
                
                stripping_command = f"bet {sig_tracker[signature]['raw_nifti']} {sig_tracker[signature]['skullstripped_nifti']} -f {skullstrip_f_val}"
                os.system(stripping_command)
                
            else:
                sig_tracker[signature]['skullstripped_nifti'] = sig_tracker[signature]['raw_nifti']
            
        # registration
        for signature, subdict in signature_relationships.items():
            if subdict['register'] == 'master':
                master_ref = sig_tracker[signature]['skullstripped_nifti']
                
                omat_path = os.path.join(processed_folder, 'master2mni.mat')
                mni_path = os.path.join(bin_folder, f'{subdict["basename"]}_mni.nii.gz')
                omat_cmd = f'flirt -in {master_ref} -ref {mni_standard} -out {mni_path} -omat omat_path'
                os.system(omat_cmd)
                
        for signature, subdict in signature_relationships.items():
            if subdict['register'] not in ('master', 'no'):
                sig_tracker[signature]['registered_nifti'] = os.path.join(bin_folder, f'{subdict["basename"]}_registered.nii.gz')
                register_command = f"flirt -in {sig_tracker[signature]['skullstripped_nifti']} -ref {master_ref} -out {sig_tracker[signature]['registered_nifti']}"
                os.system(register_command)
            else:
                sig_tracker[signature]['registered_nifti'] = sig_tracker[signature]['skullstripped_nifti']
                
        # move files to their final home :)
        for signature, subdict in signature_relationships.items():
            sig_tracker[signature]['final_nifti'] = os.path.join(processed_folder, f'{subdict["basename"]}.nii.gz')
            shutil.copyfile(sig_tracker[signature]['registered_nifti'], sig_tracker[signature]['final_nifti'])
                
        pt_status[pt]['successful'] = 1
        successful += 1
        
        
        # run FAST
        fast_folder = os.path.join(master_output_folder, 'fast')
        
        if fast_params: # if fast_params is not empty
            os.mkdir(fast_folder)
        
        for param_dict in fast_params:
            construction = f'fast -n {param_dict["n"]} -o {os.path.join(fast_folder, param_dict["baseout"])} -f {skullstrip_f_val}'
            if len(param_dict['inputs']) > 1:
                construction += f' -S {len(param_dict["inputs"])}'
            for sig in param_dict['inputs']:
                construction += f' {sig_tracker[sig]["final_nifti"]}'
                
            print(f'Construction:\n{construction}')
            os.system(construction)
            
        # run SIENA
        if run_siena:
            construction = f'sienax {sig_tracker[signature]["raw_nifti"]} -B "-f {skullstrip_f_val}"'
                
            print(f'Construction:\n{construction}')
            os.system(construction)
            
        """  
        # clean up
        # move files to their final home :)
        for signature, subdict in signature_relationships.items():
            #sig_tracker[signature]['final_nifti'] = os.path.join(master_output_folder, f'{subdict["basename"]}.nii.gz')
            final_resting_place = os.path.join(master_output_folder, f'{subdict["basename"]}.nii.gz')
            shutil.copyfile(sig_tracker[signature]['final_nifti'], final_resting_place)
               
        # delete the subfolders
            
        folder_glob = np.array(glob.glob(os.path.join(master_output_folder, '*/'))) # list of all possible subdirectories
        for f in folder_glob:
            shutil.rmtree(f)
        """


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