#!/usr/bin/env python
"""Implementation of functions for split data previously transformed to
supervised, in train, test and validation sets.

"""

def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]
