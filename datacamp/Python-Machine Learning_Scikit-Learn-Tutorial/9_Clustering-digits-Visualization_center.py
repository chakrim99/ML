#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:46:09 2018

@author: chakradhar
"""

# Import 'datasets' from sklearn
from sklearn import datasets

# Import 'train_test_split'
from sklearn.cross_validation import train_test_split

# Import 'scale'
from sklearn.preprocessing import scale

# Import 'cluster'
from sklearn import cluster
    
# Import matplotlib
import matplotlib.pyplot as plt

# Load in the 'digits' data
digits = datasets.load_digits()

# Apply 'scale' to the 'digits' data
data = scale(digits.data)

# Split the 'digits' data into training and test sets
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
        data, digits.target, digits.images, test_size = 0.25, random_state=42)

# Create KMeans model
clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)
#clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)

# Fit the training data 'X-train' to the model
clf.fit(X_train)

# Figure size in inches
fig = plt.figure(figsize=(8, 3))

# Add title
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

# For all labels (0-9)
for i in range(10):
    # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)
    # Display images
    ax.imshow(clf.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
    # Don't show the axes
    plt.axis('off')

# Show the plot
plt.show()