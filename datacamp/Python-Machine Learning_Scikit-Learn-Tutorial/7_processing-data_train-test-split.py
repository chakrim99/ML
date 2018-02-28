#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 22:19:14 2018

@author: chakradhar
"""

#Import 'datasets' from sklearn
from sklearn import datasets

# Import 'train_test_split'
from sklearn.cross_validation import train_test_split

# Import 'scale'
from sklearn.preprocessing import scale

# Import numpy
import numpy as np

# Load in the 'digits' data
digits = datasets.load_digits()

# Apply 'scale' to the 'digits' data
data = scale(digits.data)

# Split the 'digits' data into training and test sets
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, 
                                   test_size=0.25, random_state=42)

### Inspect after train and test split
# Number of training features
n_samples, n_features = X_train.shape

# Print out 'n_samples' 
print(n_samples)

# Print out 'n_features'
print(n_features)

# Number of training labels
n_digits = len(np.unique(y_train))

# Inspect 'y_train'
print(len(y_train))
