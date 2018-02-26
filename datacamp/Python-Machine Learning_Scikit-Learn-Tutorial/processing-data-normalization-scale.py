#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:55:54 2018

@author: chakradhar
"""

# Import 'datasets' from sklearn
from sklearn import datasets

# Import
from sklearn.preprocessing import scale

# Load in the 'digits' data
digits = datasets.load_digits()

# Apply 'scale' to the 'difits' data
data = scale(digits.data)