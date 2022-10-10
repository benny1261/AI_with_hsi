#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 23:00:47 2021

@author: tc_chien
"""

# importing required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
#%%

path = '/Volumes/T7 Touch/HSI-dataset/00-Analysis/07-gray-avg'
os.chdir(path)

# read the csv file using read_csv function of pandas
train = pd.read_csv('./test_2.csv')
train.head()

# reading single image using imread function of matplotlib
#image = plt.imread('./CELL/accessorial-images/cell00001-0.png')
#plt.imshow(image)

# Number of unique training images
train['filename'].nunique()

# Number of classes
train['cell_type'].value_counts()

#fig = plt.figure()

#%%
#
##add axes to the image
#ax = fig.add_axes([0,0,1,1])
#
## read and plot the image
##image = plt.imread('./CELL/accessorial-images/cell00001-0.png')
##plt.imshow(image)
#
## iterating over the image for different objects
#for _,row in train[train.filename == "cell00001.png"].iterrows():
#    xmin = row.xmin
#    xmax = row.xmax
#    ymin = row.ymin
#    ymax = row.ymax
#    
#    width = xmax - xmin
#    height = ymax - ymin
#    
#    # assign different color to different classes of objects
#    if row.cell_type == 'RBC':
#        edgecolor = 'r'
#        ax.annotate('RBC', xy=(xmax-40,ymin+20))
#    elif row.cell_type == 'WBC':
#        edgecolor = 'b'
#        ax.annotate('WBC', xy=(xmax-40,ymin+20))
#    elif row.cell_type == 'Platelets':
#        edgecolor = 'g'
#        ax.annotate('Platelets', xy=(xmax-40,ymin+20))
#        
#    # add bounding boxes to the image
#    rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = edgecolor, facecolor = 'none')
#    
#    ax.add_patch(rect)
#    
#%%

data = pd.DataFrame()
data['format'] = train['filename']

# as the images are in train_images folder, add train_images before the image name
for i in range(data.shape[0]):
    data['format'][i] = 'train_images/' + data['format'][i]

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) + ',' + str(train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + train['cell_type'][i]

os.chdir(path)
data.to_csv('annotate.txt', header=None, index=None, sep=' ')


#%%

#tt = np.load('/Users/tc_chien/Desktop/Experiment/HSI_Narlabs/Lyu/HSI-dataset/CELL/Annotations/WBC_Cancer_1/annotate1.txt')
