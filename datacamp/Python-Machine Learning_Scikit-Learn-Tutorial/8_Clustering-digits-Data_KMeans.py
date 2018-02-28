#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 21:53:23 2018

@author: chakradhar
"""

# Import 'datasets' from sklearn
from sklearn import datasets

# Import 'train_test_split'
from sklearn.cross_validation import train_test_split

# Import 'scale'
from sklearn.preprocessing import scale

# Import the 'cluster' from sklearn
from sklearn import cluster

# Load in the 'digits' data
digits = datasets.load_digits()

# Apply 'scale' to the 'digits' data
data = scale(digits.data)

# Split the 'digits' data into training and test sets
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, test_size = 0.25, random_state=42)

# Create KMeans model
clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)
#clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)

# Fit the training data 'X-train' to the model
clf.fit(X_train)
