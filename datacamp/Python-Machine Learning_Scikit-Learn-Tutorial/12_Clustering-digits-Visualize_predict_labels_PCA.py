#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 16:46:38 2018

@author: chakradhar
"""

# Import 'datasets'
from sklearn import datasets

# Import 'cluster'
from sklearn import cluster

# Import 'scale'
from sklearn.preprocessing import scale

# Import 'train test and split', 
from sklearn.cross_validation import train_test_split

# Import 'PCA'
from sklearn.decomposition import PCA

# Import 'matplotlib'
import matplotlib.pyplot as plt


# Load 'digits' data
digits = datasets.load_digits()

# Apply 'scale' to the 'digits' data
data = scale(digits.data)

# Split the data into training and test sets
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
        data, digits.target, digits.images, test_size =0.25, random_state=42)

# Create the KMeans model
clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)

# Fit the training data X_train to the model
clf.fit(X_train)

# Predict the labels for X_test
y_pred = clf.predict(X_test)

# Print out the first 100 instances of y_pred
print(y_pred[:100])

# Print out the first 100 instances of y_test
print(y_test[:100])

# Study the state of the cluster center
clf.cluster_centers_.shape

# Model and fit the 'digits' data inti the PCA model
X_PCA = PCA(n_components=2).fit_transform(X_train)

# Compute cluster centers and predict cluster index for each sample
clusters = clf.fit_predict(X_train)

# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Adjust layout
fig.suptitle('Predicted Versus Training Labels (PcA)', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots 
ax[0].scatter(X_PCA[:, 0], X_PCA[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(X_PCA[:, 0], X_PCA[:, 1], c=y_train)
ax[1].set_title('Actual Training Labels')

# Show the plots
plt.show()