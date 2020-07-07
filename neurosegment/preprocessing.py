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
import sympy as sp
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
    

'''  DEFUNCT: sympy has built in functionality to do this
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
'''


def intersection_of_plane_with_slice(slice_index, plane):
    """
    ASSUMING AXIAL ORIENTATION, finds the line that repreents the interection
    between the two planes IN 2D
    

    Parameters
    ----------
    slice_index : int
        the z index of the axial slice.
    plane : sympy Plane object
        The plane of interest.

    Returns
    -------
    sympy Line2D object.

    """

    # set up the plane of the slice
    flat_plane = sp.Plane((1,0,slice_index),(-1,0,slice_index),(0,1,slice_index))
    
    inter = plane.intersection(flat_plane)[0]
    
    intersection_2d = sp.Line(
        (inter.p1[0], inter.p1[1]),(inter.p2[0], inter.p2[1])
    )
    
    return intersection_2d


def evaluate_x_on_line(x, line):
    """
    Finds the y for a given x on a line
    

    Parameters
    ----------
    x : int or float
        x to be evaluated.
    line  sympy line2d object
        DESCRIPTION.

    Returns
    -------
    The value of y at x.

    """
    A, B, C = line.coefficients
    
    m = -A/B
    b = -C/B
    
    return m*x + b


def calculate_projected_plane_coords(slice_index, plane, x_domain = (0,500)):
    """
    Gets the coordinates the interection of a plane with your axial slice, primarily
    for plotting purposes
    

    Parameters
    ----------
    slice_index : int
        index of the axial slice
    plane : sympy Plane object
        Plane to be drawn.
    x_domain: tuple of ints or floats
        The domain to be plotted on

    Returns
    -------
    Tuple of lists containing the x and y coords of the intersecting line.

    """

    inter = intersection_of_plane_with_slice(slice_index, plane)
    intersection_2d = sp.Line(
        (inter.p1[0], inter.p1[1]),(inter.p2[0], inter.p2[1])
    )
    
    exes = np.arange(x_domain[0], x_domain[1])
    whys = [evaluate_x_on_line(x, intersection_2d) for x in exes] 
    
    
    return exes, whys


def is_partnered(coordinates, image, line):
    """
    Given a binary image and a reflecting line, checks if an input pixel has
    the same value as its reflection across the line iff the original pixel value
    is 1. If the original value is 1, always returns false
    

    Parameters
    ----------
    coordinates : tuple of ints
        coordinates of the pixel in question.
    image : 2d numpy array
        a 2d slice of an image.
    line : sympy line2d object
        a 2d sympy line.

    Returns
    -------
    A logical bit indicating if the values at the both the original and reflected
    coordinates are 1.

    """
    x, y = coordinates
    original_val = image[x,y]
    if original_val == 0:
        return 0
    original_coords= sp.Point(coordinates[0],coordinates[1])
    reflected_coords = original_coords.reflect(line)
    # not always going to be an int, need to coerce
    rx, ry = round(reflected_coords[0]), round(reflected_coords[1])

    
    try:
        reflected_val = image[rx,ry]
    except IndexError:
        return 0
    
        
    # print(f'Original: {x},{y} is {original_val}')
    # print(f'Reflected: {rx},{ry} is {reflected_val}')
    
    return int(original_val == reflected_val)


def score_midsagittal(image, plane, n_slices=None):
    """
    Scores the quality of an approximated midsagittal plane based on symmetry of
    the BINARY-ized Sobel-filtered scan
    

    Parameters
    ----------
    image : 3d numpy array
        An axial scan.
    plane : sympy plane object
        The plane that approximates the midsagittal plane.
    n_slices: int
        The number of slices to actually use to calculate the score. Must be
        equal to or less than the number of slices in the image. Used to reduce
        runetime. If None, all slices will be used

    Returns
    -------
    A score as a float between 0 and 1, where 1 is perfect.

    """
    
    num_z_levels = image.shape[2]

    n_edges = 0 # number of voxels with value of 1
    n_paired = 0
    
    for z in range(num_z_levels):
        sub_image = image[:,:,z]
        n_edges += sum(sum(sub_image))
        reflecting_line = intersection_of_plane_with_slice(z, plane)
        print(f'On level --{z}-- ({sum(sum(sub_image))} pixels to check)')
        
        scoreboard = np.zeros((image.shape[0], image.shape[1]))
        for x in range( image.shape[0]):
            for y in range(image.shape[1]):
                scoreboard[x,y] = is_partnered((x,y), sub_image, reflecting_line)
        n_paired += sum(sum(scoreboard))
        
    return n_paired / n_edges
        
    

    
    
    
    
    
    
    


