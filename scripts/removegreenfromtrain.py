#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 22:03:44 2017

@author: chris
"""

import os
import matplotlib.pyplot as plt
import skimage.io as sio


image_paths = []
to_remove = []

for t in ['Type_1', 'Type_2', 'Type_3']:
    for filename in os.listdir('../data/images/pre/train/' + t):
        image_paths.append('../data/images/pre/train/' + t + '/' + filename)

# Finds all the green images in the traindata.
for imagep in image_paths:
    a = sio.imread(imagep)
    if a[:,:,1].sum() > (a[:,:,0].sum() + a[:,:,2].sum()) * 10:
        # Plot if you want to confirm that the process went well
        '''
        plt.imshow(a)
        plt.show()
        '''
        to_remove.append(imagep)

for filename in to_remove:
    os.remove(filename)