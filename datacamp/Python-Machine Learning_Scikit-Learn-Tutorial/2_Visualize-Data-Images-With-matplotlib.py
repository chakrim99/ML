#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 14:42:32 2018

@author: chakradhar
"""

# Import matplotlib
import matplotlib.pyplot as plt

# Import `datasets` from `sklearn`
from sklearn import datasets

# Load in the `digits` data
digits = datasets.load_digits()

# Set up with figure with figure size (width, height) in inches. This will create a blank canvas
fig = plt.figure(figsize=(6,6))

# Adjust the subplots
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# for each of the 64 images
for i in range(64):
    # Initialize the sub-plots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    # Display an image at the i-th position. Color map with binary colors, and will result in black, grey and white
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))
    
    
# Show the plot
plt.show()    
    