#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for neurosegment

"""


import sympy as sp
import matplotlib.pyplot as plt

from preprocessing import read_nifti, skull_strip, sobelize, binary_by_percentile_threshold
from preprocessing import calculate_projected_plane_coords, is_partnered, intersection_of_plane_with_slice
from preprocessing import score_midsagittal


px, py = (238, 242)

arbitrary_plane = sp.Plane((0,0,10),(300,300,20),(300,300,10))
slice_num = 23
img_path = r'/Users/manusdonahue/Documents/Sky/Infarcts/Donahue_114402.XMLPARREC.dcm2niix/NIFTI/Donahue_114402.04.01.14-42-52.WIP_MJD_FLAIR_AX_3MM_SENSE.01.nii'


img = read_nifti(img_path)
stripped_img, mask = skull_strip(img)
sobel_img = sobelize(stripped_img)
edge_img, abs_thresh = binary_by_percentile_threshold(sobel_img, 3, invert=True)


# note that when plotting with imshow, imaging conventions for coordinates are used
# essentially x and y are swapped and the origin is at the upper left corner

raw_data = img[:,:,slice_num]
fig, ax = plt.subplots(2,2)
ax[0][0].imshow(raw_data, interpolation='nearest', cmap='gray')

stripped_data = stripped_img[:,:,slice_num]
ax[1][0].imshow(stripped_data, interpolation='nearest', cmap='gray')

sobel_data = sobel_img[:,:,slice_num]
ax[0][1].imshow(sobel_data, interpolation='nearest', cmap='gray')

edge_data = edge_img[:,:,slice_num]
ax[1][1].imshow(edge_data, interpolation='nearest', cmap='gray')

exes, whys = calculate_projected_plane_coords(slice_num, arbitrary_plane)
plt.plot(whys,exes)


the_line = intersection_of_plane_with_slice(slice_num, arbitrary_plane)
reflected = sp.Point(px,py).reflect(the_line)
rx, ry = reflected[0], reflected[1]


plt.scatter(py, px, color='red')
plt.scatter(ry, rx, color='blue')

plt.show()

isp = is_partnered((px,py), edge_data, the_line)
print(isp)

sc = score_midsagittal(edge_img, arbitrary_plane, None)

