#!/usr/bin/env python
"""Implementation of correlaton methods

sources:
- 'https://realpython.com/numpy-scipy-pandas-correlation-python/'
"""
import numpy as np


def pearson_correlation(x, y):
    '''
        This function implement Pearson correlation.
        source: numpy.corrcoef
        ----------
        input
        -----
        x: array_like
        y: array_like        
        Output
        ------
        ndarray, the correlation coefficient matrix of the variables.
    '''
    return np.corrcoef(x, y)

