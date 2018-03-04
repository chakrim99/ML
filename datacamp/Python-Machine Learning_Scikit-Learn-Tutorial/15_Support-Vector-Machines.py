#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 17:30:01 2018

@author: chakradhar
"""

# Import 'datasets'
from sklearn import datasets

# Import the 'svm' (Support Vector Machines) model
from sklearn import svm

# Import 'scale'
from sklearn.preprocessing import scale

# Import 'train test and split', 
from sklearn.cross_validation import train_test_split

# Load 'digits' data
digits = datasets.load_digits()

# Apply 'scale' to the 'digits' data
data = scale(digits.data)

# Split the data into training and test sets
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
        data, digits.target, digits.images, test_size =0.25, random_state=42)

# Create the SVC model
svc_model = svm.SVC(gamma=0.001, C=100., kernel='linear')

# Fit the data to the SVC model
svc_model.fit(X_train, y_train)