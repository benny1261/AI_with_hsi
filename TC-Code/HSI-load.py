#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:40:01 2021

@author: tc_chien
"""

import os
from spectral.io import envi
import spectral
import cv2
import numpy as np

name = "00_processed"

path = os.path.dirname(os.path.dirname(__file__))+r'/data'
os.chdir(path)

spectral.settings.envi_support_nonlowercase_params = 'TRUE'

img = envi.open(name+ ".hdr" , name+ ".raw")        # our hsi metadata stored in ENVI raster format

print('===============================')
arr = img.load()
print(arr.info())
print("Shape:", arr.shape)

length = arr.shape[0] #1088
width = arr.shape[1] #2048
height = arr.shape[2] #150

arr = arr.copy()

#Defective pixel
arr[arr < -10]=0
arr[arr > 10]=0
dp = np.where(arr == 0) 
dp = np.asarray(dp)
#np.savetxt('DP.csv', dp, fmt="%d",delimiter=",")

#Defective pixel correct
dpc1 = np.mean(arr)

arr = np.array(arr,dtype='float16')

for i in range(dp.shape[1]):
    x = dp[0,i]
    y = dp[1,i]
    z = dp[2,i]

    if x == 0:
        arr[x,y,z] = dpc1
        continue
    if x+1 == 1088:
        arr[x,y,z] = dpc1
        continue
    if y == 0:
        arr[x,y,z] = dpc1
        continue
    if y+1 == 2048:
        arr[x,y,z] = dpc1
        continue

    dpc2 = (arr[x+1,y,z] + arr[x-1,y,z] + arr[x,y+1,z] + arr[x,y-1,z])/4

    arr[x,y,z] = dpc2

img_gray = cv2.normalize(np.float32(arr[:,:,20]), None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC3)
cv2.imwrite(name+'.png', img_gray)