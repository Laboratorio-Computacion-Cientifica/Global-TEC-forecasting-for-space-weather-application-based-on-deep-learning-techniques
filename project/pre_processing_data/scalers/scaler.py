#!/usr/bin/env python
"""Implementation of scalers functions

The basic idea of this module, is to implement:
    - Normalization -->
            y = (x – min) / (max – min)

sources:
- 'https://machinelearningmastery.com/how-to-improve-neural-network-stability
-and-modeling-performance-with-data-scaling/'
"""
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def scale_data(data, method='normalization'):
    '''
        This function implement normalization of data.
        ----------
        input
        -----
        data: numpy.array
        method: normalization or standardization. default normalization
        Output
        ------
        scaledData: data scaled
        scaler: scaler fited for data
    '''
    # define scaler
    if(method == 'normalization'):
        # define min max scaler
        scaler = MinMaxScaler()
    elif(method =='standardization'):
        # define standard scaler
        scaler = StandardScaler()
    # fit scaler
    scaler.fit(data)
    # transform data
    scaledData = scaler.transform(data)
    scaledData = np.round(scaledData,2)
    # return scaled data and scaler
    return scaledData, scaler



def unscale_data(data, scaler):
    '''
        This function implement the unscale of data.
        ----------
        input
        -----
        data: numpy.array
        scaler: scaler fited for main data
        Output
        ------
        array.numpy denormalizedunscaled
    '''
    return scaler.inverse_transform(data)
