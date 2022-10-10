#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 01:03:02 2021

@author: tc_chien
"""

import os
import cv2
import numpy as np
from glob import glob

#%%

path = '/Volumes/T7 Touch/HSI-dataset/00-Analysis/07-gray-avg'

#os.chdir(path)

if os.path.exists(path+'/augment'):
    print ("Folder exist")
else:
    os.mkdir(path+'/augment')#建立資料夾
    print ("Folder is created")

#%%

names = glob(path + '/*.png')

os.chdir(path+'/augment')

for n in names:
    img = cv2.imread(n)
    
    width = img.shape[0]
    length = img.shape[1]
    
    img5 = cv2.resize(img, (length, int(width*0.625)), interpolation=cv2.INTER_LINEAR)
    img6 = cv2.resize(img, (length, int(width*0.75)), interpolation=cv2.INTER_LINEAR)
    img7 = cv2.resize(img, (length, int(width*0.875)), interpolation=cv2.INTER_LINEAR)
    
    img_5 = cv2.resize(img, (int(length*0.625), width), interpolation=cv2.INTER_LINEAR)
    img_6 = cv2.resize(img, (int(length*0.75), width), interpolation=cv2.INTER_LINEAR)
    img_7 = cv2.resize(img, (int(length*0.875), width), interpolation=cv2.INTER_LINEAR)
    
    name = n.split('/')[-1]
    name0 = name.split('.')[0] + '-o.png'
    name5 = name.split('.')[0] + '-w5.png'
    name6 = name.split('.')[0] + '-w6.png'
    name7 = name.split('.')[0] + '-w7.png'
    
    name_5 = name.split('.')[0] + '-l5.png'
    name_6 = name.split('.')[0] + '-l6.png'
    name_7 = name.split('.')[0] + '-l7.png'
    
    cv2.imwrite(name0, img)
    cv2.imwrite(name5, img5)
    cv2.imwrite(name6, img6)
    cv2.imwrite(name7, img7)
    cv2.imwrite(name_5, img_5)
    cv2.imwrite(name_6, img_6)
    cv2.imwrite(name_7, img_7)

