import os
from spectral.io import envi
import spectral
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from sklearn.decomposition import IncrementalPCA
#from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

import random

#%%--------設定參數-------------------------------------------------------------
cellname = 'MCF7_3'

n_components = 15 #PCA降維
clusters_number = 9

#colormap = ['red','orange','yellow','green','blue','cadetblue','purple','black','darkgray','deeppink','firebrick','orangered','lawngreen','cyan','chocolate','navy']
colormap = ['red','orangered','coral','lightcoral','olive','olivedrab','yellowgreen','greenyellow','steelblue','dodgerblue','lightskyblue','cyan','purple','blueviolet','plum','deeppink']
cm = np.loadtxt(open("/Users/tc_chien/Desktop/Experiment/HSI_Narlabs/PythonCode/colormap2.csv"),delimiter=",",skiprows=0, dtype='int')

#wavelength = np.loadtxt('/Users/tc_chien/Desktop/Experiment/HSI_Narlabs/Lyu/wavelength_900.csv',dtype=float)
wavelength = np.genfromtxt('/Users/tc_chien/Desktop/Experiment/HSI_Narlabs/PythonCode/wavelength_900.csv',dtype=float,skip_header=0)
wavelength[0]=470.001

path = '/Volumes/T7 Touch/HSI-dataset/MCF7/'+cellname#指定路徑


if os.path.exists(path+'/cropimg'):
    print ("Folder exist")
else:
    os.mkdir(path+'/cropimg')#建立資料夾
    print ("Folder is created")

#%%----------------------------------------------------------------------1mins

os.chdir(path)

spectral.settings.envi_support_nonlowercase_params = 'TRUE'

img = envi.open(cellname+".hdr" , cellname+".raw")

print(img.__class__)
print(img)
print('===============================')

arr = img.load()
arr.__class__
print(arr.info())
print("Shape:")
print(arr.shape)

length = arr.shape[0] #1088
width = arr.shape[1] #2048
height = arr.shape[2] #150

metadata =img.metadata
arr = arr[:,:,0:height] #只取400-700mm

os.chdir(path+'/cropimg')#讀取資料夾

#Defective pixel
arr[arr < -10]=0
arr[arr > 10]=0
dp = np.where(arr == 0) 
dp = np.asarray(dp)
np.savetxt('DP.csv', dp, fmt="%d",delimiter=",")

#Defective pixel correct
#dpc = np.mean(arr)
#arr[arr == 0]= dpc

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
cv2.imwrite('img_gray.png', img_gray)

arr = np.array(arr,dtype='float16')

arr = arr/(np.mean(arr))

#%% singlecell-crop

os.chdir(path)
scp = np.loadtxt(open("scp.csv"),delimiter=",",skiprows=0, dtype='int') #single cell position
#randomlist = random.sample(range(0, scp.shape[0]-1), 9)
#scp = scp[[randomlist],:]
#scp = scp.reshape(len(randomlist),4)

os.chdir(path+'/cropimg')

for i in range(scp.shape[0]):
    xmin = scp[i,0];xmax = scp[i,2] #2048
    ymin = scp[i,1];ymax = scp[i,3] #1088
    sc_hsi = arr[ymin:ymax,xmin:xmax,:]
    sc_gray = cv2.normalize(np.float32(sc_hsi[:,:,20]), None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC3)
    
    np.save(cellname+'_'+str(i+1), sc_hsi)
    cv2.imwrite(cellname+'_'+str(i+1)+'.png', sc_gray)
