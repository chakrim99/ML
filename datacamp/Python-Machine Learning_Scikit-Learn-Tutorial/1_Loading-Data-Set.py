#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:35:21 2018

@author: chakradhar
"""

# Import `datasets` from `sklearn`
from sklearn import datasets

# Import numpy
import numpy as np

# Load in the `digits` data
digits = datasets.load_digits()

# Print the `digits` data 
print(digits)

# Get the keys of the `digits` data
print(digits.keys())

# Print out the data
print(digits.data)

# Print out the target values
print(digits.target)

# Print out the description of the `digits` data
print(digits.DESCR)

# Isolate the `digits` data
digits_data = digits.data

# Inspect the shape
print(digits_data.shape)

# Isolate the target values with `target`
digits_target = digits.target

# Inspect the shape
print(digits_target.shape)

# Print the number of unique labels
number_digits = len(np.unique(digits.target))

# Isolate the `images`
digits_images = digits.images

# Inspect the shape
print(digits_images.shape)

# Whether all array elements along a given axis evaluate to True
print(np.all(digits.images.reshape((1797,64)) == digits.data))