#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:50:27 2020

@author: manusdonahue
"""

import numpy as np

def is_empty(x):
    """
    Like np.isnan, but won't throw an error on strings
    

    Parameters
    ----------
    x : anything
        variable you're checking.

    Returns
    -------
    See np.isnan

    """
    
    try:
        return np.isnan(x)
    except TypeError:
        if type(x) is str:
            return False
        else:
            raise TypeError
            
def numbery_string_to_number(x):
    """
    Take a string that has a number in it and returns only the numbers
    """
    if type(x) is not str:
        return x
    
    a = [int(i) for i in x.split() if i.isdigit()]
    if len(a) == 1:
        return a[0]
    elif len(a) == 0:
        return None
    else:
        return(a)