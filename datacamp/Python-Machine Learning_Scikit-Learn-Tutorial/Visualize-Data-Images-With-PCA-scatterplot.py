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

# Import matplotlib
import matplotlib.pyplot as plt

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

colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

for i in range(len(colors)):
    x = reduced_data_rpca[:, 0][digits.target == i]
    y = reduced_data_rpca[:, 1][digits.target == i]
    plt.scatter(x,y, c=colors[i])
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
plt.show()
