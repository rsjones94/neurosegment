#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for preprocessing cranial MRI imagery, namely skull stripping and
finding the midsaggital plane

Note that deepbrain requires use of a 1.X version of tensorflow - does not work
with v2.X even though pip installing deepbrain will install the latest version
of tensorflow with itS
"""


import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from deepbrain import Extractor

slice_num = 23
img_path = r'/Users/manusdonahue/Documents/Sky/Infarcts/Donahue_114402.XMLPARREC.dcm2niix/NIFTI/Donahue_114402.04.01.14-42-52.WIP_MJD_FLAIR_AX_3MM_SENSE.01.nii'

# Load a nifti as 3d numpy image [H, W, D]
img = nib.load(img_path).get_fdata()

ext = Extractor()

# `prob` will be a 3d numpy image containing probability 
# of being brain tissue for each of the voxels in `img`
prob = ext.run(img) 

# mask can be obtained as:
mask = prob > 0.5

stripped_img = img*mask


raw_data = img[:,:,slice_num]
fig, ax = plt.subplots(1,2)
ax[0].imshow(raw_data, interpolation='nearest', cmap='gray')

stripped_data = stripped_img[:,:,slice_num]
ax[1].imshow(stripped_data, interpolation='nearest', cmap='gray')
plt.show()