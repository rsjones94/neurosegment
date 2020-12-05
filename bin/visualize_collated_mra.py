#!/usr/bin/env python

import os
import sys
import getopt
import glob
import shutil
from datetime import datetime
from contextlib import contextmanager
from time import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import nibabel as nib


data_master = '/Users/manusdonahue/Documents/Sky/nigeria_mra/data/'
target = '/Users/manusdonahue/Documents/Sky/nigeria_mra/vis/'



##### 


folder_glob = np.array(glob.glob(os.path.join(data_master, '*/'))) # list of all possible subdirectories

for f in folder_glob:
    mr_id = os.path.basename(os.path.normpath(f))
    print(f'On {mr_id}')
    fig_name = os.path.join(target,f'{mr_id}.png')

    head_mra_name = os.path.join(f,'headMRA.nii.gz')
    head_mra_mip_name = os.path.join(f,'headMRA_mip.nii.gz')
    
    names = [head_mra_name, head_mra_mip_name]
    titles = ['Head MRA', 'Head MRA MIP']
    
    fig, ax = plt.subplots(1,2)
    
    for n,t,a in zip(names,titles,ax):
        a.set_title(t)
        a.axis('off')
        try:
            data = nib.load(n)
            mat = data.get_fdata()
            dims = mat.shape
            z = dims[2]
            half = int(z/2)
            
            sli = np.rot90(mat[:,:,half].T,2)
            a.imshow(sli, cmap=matplotlib.cm.gray)
        except FileNotFoundError:
            print(f'File {n} does not exist')
    plt.tight_layout()
    fig.savefig(fig_name)
        
    