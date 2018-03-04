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

# Import `metrics`
from sklearn import metrics

# Import 'scale'
from sklearn.preprocessing import scale

# Import 'train test and split', 
from sklearn.cross_validation import train_test_split

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

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

# Set the parameter candidates
parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

# Create a classifier with the parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)

# Train the classifier on training data
clf.fit(X_train, y_train)

# Apply the classifier to the test data, and view the accuracy score
clf.score(X_test, y_test)  

# Train and score a new classifier with the grid search parameters
svm.SVC(C=10, kernel='rbf', gamma=0.001).fit(X_train, y_train).score(X_test, y_test)

# Assign the predicted values to `predicted`
predicted = svc_model.predict(X_test)

# Print the classification report of `y_test` and `predicted`
print(metrics.classification_report(y_test, predicted))

# Print the confusion matrix of `y_test` and `predicted`
print(metrics.confusion_matrix(y_test, predicted))