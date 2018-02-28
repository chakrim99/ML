#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:24:15 2018

@author: chakradhar
"""

# Import `datasets` from `sklearn`
from sklearn import datasets

# Import 'RandomizePCA' from 'sklearn.decomposistion'
from sklearn.decomposition import RandomizedPCA

# Import 'PCA' from 'sklearn.decomposistion'
from sklearn.decomposition import PCA


# Load in the `digits` data
digits = datasets.load_digits()

# Create a Randomized PCA model that takes two components
randomized_pca = RandomizedPCA(n_components=2)

# Fit and transform the data to the model
reduced_data_rpca = randomized_pca.fit_transform(digits.data)

# Create a regular PCA model
pca = PCA(n_components=2)

# Fit and transform the data to the model
reduced_data_pca = pca.fit_transform(digits.data)

# Inspect the shape
reduced_data_rpca.shape

# Print out the data
#print(reduced_data_rpca)
#print(reduced_data_pca)


