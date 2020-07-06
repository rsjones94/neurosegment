#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for preprocessing cranial MRI imagery, namely skull stripping and
finding the midsaggital plane

Note that deepbrain requires use of a 1.X version of tensorflow - does not work
with v2.X even though pip installing deepbrain will install the latest version
of tensorflow with itS

The procedure for extracting the midsagittal plane is based on the paper
'A new symmetry-based method for mid-sagittal plane extraction in neuroimages'
(https://ieeexplore.ieee.org/document/5872407)
"""

import numpy as np
from skimage import img_as_float64
from scipy import ndimage
import nibabel as nib
from deepbrain import Extractor


def read_nifti(img_path):
    """
    Wrapper for nibabel to read in NIfTI scans.
    

    Parameters
    ----------
    img_path : string
        The path to the .nii scan
            
    Returns
    -------
    A numpy array representing the scan

    """
    img = nib.load(img_path).get_fdata()
    
    return img
    

def skull_strip(img):
    """
    Removes non-brain voxels from a scan
    

    Parameters
    ----------
    img : numpy array
        3d np array representing the scan

    Returns
    -------
    A tuple containing the skull-stripped image followed by the binary mask
    used to strip the original image

    """
    ext = Extractor()

    # `prob` will be a 3d numpy image containing probability 
    # of being brain tissue for each of the voxels in `img`
    prob = ext.run(img) 
    # mask can be obtained as:
    mask = prob > 0.5
    
    stripped_img = img*mask
    
    return stripped_img, mask


def sobelize(img):
    """
    Wrapper to apply a 3d Sobel operator to the img

    Parameters
    ----------
    img : numpy array
        3d np array representing the image.

    Returns
    -------
    numpy array of the image as processed by the Sobel operator.

    """
    sobeled = ndimage.sobel(img, axis=-1)
    
    return sobeled


def binary_by_percentile_threshold(img, threshold=95, invert=False):
    """
    Takes an array and sets pixels to 1 or 0 depending on if they exceed a
    threshold value as defined by the value at a given threshold percentile.
    
    According to the midsag paper referenced, the brightest 5% of pixels
    in the Sobel-filtered image should approximate edges. However,
    it appears in FLAIR imagery that it is actually the DARKEST 5% of pixels,
    so if using on FLAIR imagery set threshold to 5 and invert to True
    

    Parameters
    ----------
    img : numpy array
        3d np array representing the image.
    threshold : float, optional
        The percentile to threshold the pixels values at (inclusive).
        The default is 95.
    invert : bool, optional
        By default, pixels above the threshold are set to 1 and those not
        exceeding the threshold are set to 0. If invert is True, this scheme is
        inverted. The default is False.

    Returns
    -------
    A tuple of the binary-ified array followed by the abolute value of the
    threshold.

    """
    
    absolute_thresh = np.percentile(img, threshold)
    
    if invert:
        filtered = img<= absolute_thresh
    else:
        filtered = img >= absolute_thresh
    
    return filtered.astype(int), absolute_thresh
    
    
def reflect_across_line(coords, line):
    """
    Finds the coordinates of a 2d point reflected across a given line.
    
    https://mathteachersresource.com/assets/reflection-over-any-oblique-line-pdf.pdf
    

    Parameters
    ----------
    coords : tuple of ints or floats
        A tuple of the form (p,q) giving the coordinates to be reflected.
    line : tuple of ints or floats
        A tuple of the form (m,b) giving the slope and intercept (y=mx+b)
        of the line to be reflected across.

    Returns
    -------
    The new coordinates (r,s) as tuple.

    """
    
    m = line[0]
    b = line[1]
    
    p = coords[0]
    q = coords[1]
    
    coef = 1 / (m**2 + 1)
    matrix = np.array([[1-m**2,2*m,-2*b*m],[2*m, m**2-1, 2*b],[0,0,1]])
    vector = np.array([[p],[q],[1]])
    
    result = coef*np.matmul(matrix,vector)
    
    r = result[0][0]
    q = result[1][0]
    
    return r,q
    
    
    
    
    
    
    


