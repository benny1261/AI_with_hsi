import os
from spectral.io import envi
import spectral
import cv2
import numpy as np
    
os.chdir('/Volumes/T7 Touch/HSI-dataset/WBC_HCT116_MCF7')
cm = np.loadtxt(open("/Users/tc_chien/Desktop/Experiment/HSI_Narlabs/PythonCode/colormap2.csv"),delimiter=",",skiprows=0, dtype='int')

ref = np.genfromtxt('/Volumes/T7 Touch/HSI-dataset/MCF7/MCF7_3/wholeimg_/6_cluster_avg.csv',dtype=float, delimiter=',')
arr = np.load('/Volumes/T7 Touch/HSI-dataset/WBC_HCT116_MCF7/WBC_HCT116_MCF7.npy')

img_gray = cv2.imread('/Volumes/T7 Touch/HSI-dataset/WBC_HCT116_MCF7/img_gray_1.png')

length = arr.shape[0] 
width = arr.shape[1] 
height = arr.shape[2] 

dist = np.zeros((length,width), dtype='float64', order='C')
for i in range(length):
    for j in range(width):
        d = np.linalg.norm(ref[:,3]-arr[i,j,:])
        dist[i,j] = d

thr = 0.1
thr_v = thr*(np.max(dist)-np.min(dist)) + np.min(dist) #thr = (thr_v-最小值)/(最大值－最小值)

dist_thr = np.where(dist <= thr_v)
dist_thr = np.asarray(dist_thr)

result = np.zeros((length,width),dtype='int')

for i in range(dist_thr.shape[1]):
    result[dist_thr[0,i],dist_thr[1,i]] = 1
    
img_r = result*cm[9][0]
img_g = result*cm[9][1]
img_b = result*cm[9][2]

img_rgb = cv2.merge([img_b,img_g,img_r]).astype(np.uint8)#合併B、G、R分量
merged = cv2.addWeighted(img_gray, 1, img_rgb, 1, 0)
cv2.imwrite('ttt_dist_'+str(thr)+'.png', merged)




