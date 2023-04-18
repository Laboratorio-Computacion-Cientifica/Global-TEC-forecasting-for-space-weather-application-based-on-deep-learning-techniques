#!/usr/bin/env python
"""Implementation of functions for split data for multivariate cases

The basic idea of this module, is to implement:
- Multivariate Multi-step LSTM Models
    - Multiple Input Multi-step Output.
    - Multiple Parallel Input and Multi-step Output.
"""

from numpy import array

def sliding_window(sequences, nStepsIn, nStepsOut):
    '''
        This function implement the split for Multiple Input Multi-step Output.
        ----------
        input
        -----
        sequences: stack arrays in sequence horizontally (numpy.hstack).
        nStepsIn: number of input time steps (int).
        nStepsOut: number of output time steps (int).
        Output
        ------
        X: numpy.ndarray
        y: numpy.ndarray
    '''    
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + nStepsIn
        out_end_ix = end_ix + nStepsOut-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]        
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


