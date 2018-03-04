#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 17:07:20 2018

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

from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score

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

# Study the state of the cluster center
clf.cluster_centers_.shape

# Print
print('% 9s' % 'inertia homo    compl   v-means ARI     AMI      silhoutte')

# Print
print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          %(clf.inertia_,
      homogeneity_score(y_test, y_pred),
      completeness_score(y_test, y_pred),
      v_measure_score(y_test, y_pred),
      adjusted_rand_score(y_test, y_pred),
      adjusted_mutual_info_score(y_test, y_pred),
      silhouette_score(X_test, y_pred, metric='euclidean')))