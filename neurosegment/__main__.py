#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for neurosegment

"""

import matplotlib.pyplot as plt

from preprocessing import read_nifti, skull_strip, sobelize, binary_by_percentile_threshold


slice_num = 23
img_path = r'/Users/manusdonahue/Documents/Sky/Infarcts/Donahue_114402.XMLPARREC.dcm2niix/NIFTI/Donahue_114402.04.01.14-42-52.WIP_MJD_FLAIR_AX_3MM_SENSE.01.nii'


img = read_nifti(img_path)
stripped_img, mask = skull_strip(img)
sobel_img = sobelize(stripped_img)
edge_img, abs_thresh = binary_by_percentile_threshold(sobel_img, 3, invert=True)


raw_data = img[:,:,slice_num]
fig, ax = plt.subplots(2,2)
ax[0][0].imshow(raw_data, interpolation='nearest', cmap='gray')

stripped_data = stripped_img[:,:,slice_num]
ax[1][0].imshow(stripped_data, interpolation='nearest', cmap='gray')

sobel_data = sobel_img[:,:,slice_num]
ax[0][1].imshow(sobel_data, interpolation='nearest', cmap='gray')

edge_data = edge_img[:,:,slice_num]
ax[1][1].imshow(edge_data, interpolation='nearest', cmap='gray')

plt.show()

