#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 23:17:24 2021

@author: tc_chien
"""

import os
import cv2
import numpy as np
from glob import glob
    
#%%
path = '/Users/tc_chien/Desktop/碩班/07-畢業論文/202107畢業口試/圖/results_imgs-fp-mappen-test'

if os.path.exists(path+'/avg_light'):
    print ("Folder exist")
else:
    os.mkdir(path+'/avg_light')#建立資料夾
    print ("Folder is created")

total = glob(path + '/*.png')

for n in total:
            
    name = n.split('/')[-1]
    name_ = name.split('.')[0] + '.png'   
    img = cv2.imread(n, 0)
    
    m = np.mean(img)
    img = img*(100/m)
    
    os.chdir(path + '/avg_light')
    print(np.mean(img))
    
    cv2.imwrite(name_, img)
            
#%%






