#!/usr/bin/env python
"""Implementation of functions for split data previously transformed to
supervised, in train, test and validation sets.

"""

def train_test_split(data, n_test, stepsIn):
    amount = int(n_test*len(data))
    while(amount%stepsIn):
        amount -= 1
    print("this amount", amount)
    return data[:-amount], data[-amount:], amount
