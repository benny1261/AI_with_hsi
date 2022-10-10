# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:29:25 2020

@author: GameToGo
"""
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

#%%--------設定參數-------------------------------------------------------------
cellname = 'WBC_MCF7_1'

n_components = 15 #PCA降維
clusters_number = 9

#colormap = ['red','orange','yellow','green','blue','cadetblue','purple','black','darkgray','deeppink','firebrick','orangered','lawngreen','cyan','chocolate','navy']
colormap = ['red','orangered','coral','lightcoral','olive','olivedrab','yellowgreen','greenyellow','steelblue','dodgerblue','lightskyblue','cyan','purple','blueviolet','plum','deeppink']
cm = np.loadtxt(open("/Users/tc_chien/Desktop/Experiment/HSI_Narlabs/PythonCode/colormap2.csv"),delimiter=",",skiprows=0, dtype='int')

#wavelength = np.loadtxt('/Users/tc_chien/Desktop/Experiment/HSI_Narlabs/Lyu/wavelength_900.csv',dtype=float)
wavelength = np.genfromtxt('/Users/tc_chien/Desktop/Experiment/HSI_Narlabs/PythonCode/wavelength_900.csv',dtype=float,skip_header=0)
wavelength[0]=470.001

path = '/Volumes/T7 Touch/HSI-dataset/WBC_MCF7/'+cellname#指定路徑


if os.path.exists(path+'/wholeimg'):
    print ("Folder exist")
else:
    os.mkdir(path+'/wholeimg')#建立資料夾
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

os.chdir(path+'/wholeimg')#讀取資料夾

#Defective pixel
arr[arr < -10]=0
arr[arr > 10]=0
dp = np.where(arr == 0) 
dp = np.asarray(dp)
np.savetxt('DP.csv', dp, fmt="%d",delimiter=",")

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

#arr = np.load('/Users/tc_chien/Desktop/Experiment/HSI_Narlabs/Lyu/210312/WBC_HCT116/WBC_HCT116_2.npy')

#%%

length = arr.shape[0] #1088
width = arr.shape[1] #2048
height = arr.shape[2] #150

img_gray = cv2.normalize(np.float32(arr[:,:,20]), None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC3)
cv2.imwrite('img_gray.png', img_gray)


#%%----------flatten-----------------------------------------------------10min
f_arr = np.zeros((length*width,height), dtype='float16', order='C')
for i in range(length):
    for j in range(width):
        
        f_arr[width * i + j, :] = arr[i,j, :]
        
#----------IPCA降維---------------------

ipca = IncrementalPCA(n_components=n_components, batch_size=15)
arr_ipca = ipca.fit_transform(f_arr)


#----------kmeans分群----------------------------------------------------------

kmeans = KMeans(n_clusters = clusters_number)
kmeans.fit(arr_ipca)
y_kmeans = kmeans.predict(arr_ipca)

#-----------回復成xy------------------------------------------------------------

y_arr = np.reshape(y_kmeans, (length,width))
y_arr_ipca = np.reshape(arr_ipca, (length,width,arr_ipca.shape[1]))

cluster_result = cv2.normalize(y_arr, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC3)

#cluster position
np.savetxt(str(clusters_number)+'_cluster_result.csv', y_arr, fmt="%d", delimiter=",")
#after ipca result
np.save(str(clusters_number)+'_ipca_result', y_arr_ipca)


cv2.imwrite(str(clusters_number)+'_cluster_result.png', cluster_result)


#%%--------分別取出每個cluster的位置，存成新的矩陣----------------------------6mins

cluster_points = []
for k in range(clusters_number):
    cluster_point = []
    #result = np.zeros((1216,1936),dtype='int')
    for i in range(length):
        for j in range(width):
            if y_arr[i][j] == k :
                #result[i][j] = k
                cluster_point.append([i,j])
    cluster_point = np.asarray(cluster_point)
    cluster_points.append(cluster_point)
cluster_points = np.asarray(cluster_points)
 #np.save("points.npy",cluster_points)

#---------將位置帶回高光譜-並依cluster計算平均值，畫出curve------------------------

std_cluster = []
avg_cluster = []
for k in range(clusters_number):
    
    cluster_point = cluster_points[k]
    hsi_boc = np.zeros((cluster_point.shape[0],height),dtype='float64') #hsi based on cluster
    
    for i in range(cluster_point.shape[0]):
        
        hsi_boc[i,:] = arr[cluster_point[i][0],cluster_point[i][1],:]
    
    std = []
    avg = []
    for j in range(height):
        std.append(np.std(hsi_boc[:,j]))
        avg.append(np.mean(hsi_boc[:,j]))
        
    std = np.asarray(std)
    avg = np.asarray(avg)
    std_cluster.append(std)
    avg_cluster.append(avg)
    
#    thre_3 = avg + std + std + std
#    
#    for l in range(height):
#        #tp1 = np.where(hsi_boc[:,l] >= thre_1[l,i])
#        tp3 = np.where(hsi_boc[:,l] >= thre_3[l])
#        tp3 = np.asarray(tp3)
#        tp3 = np.reshape(tp3, tp3.shape[1])
#        
#        print('thre3:')
#        print(tp3)
    
    hsi_boc = hsi_boc.astype('float64')
    #np.save('cluster' + str(k+1), hsi_boc)
    
std_cluster = np.asarray(std_cluster)
avg_cluster = np.asarray(avg_cluster)

cluster_std = std_cluster.T
cluster_avg = avg_cluster.T

np.savetxt(str(clusters_number)+'_cluster_avg.csv', cluster_avg, fmt="%f", delimiter=",")
np.savetxt(str(clusters_number)+'_cluster_std.csv', cluster_std, fmt="%f", delimiter=",")


#%%------利用每個cluster對應位置，套色疊圖--------------------------------------4mins

#os.chdir(path)#指定路徑
#img_gray = cv2.imread(cellname+'.png')
os.chdir(path+'/wholeimg')#指定路徑

img_gray = cv2.imread('img_gray.png')

y_arr = y_arr + 1 #把所有元素都加1，避免0出現，帶進去全零矩陣時看不見
img = np.zeros((length,width,3), dtype='uint8', order='C')

for k in range(1,1+clusters_number):
    
    result = np.zeros((length,width),dtype='int')
    for i in range(length):
        for j in range(width):
            if y_arr[i][j] == k:
                result[i][j] = y_arr[i][j]   
    result = result/(k)
    img_r = result*cm[k-1][0]
    img_g = result*cm[k-1][1]
    img_b = result*cm[k-1][2]
    
    img_rgb = cv2.merge([img_b,img_g,img_r]).astype(np.uint8)#合併B、G、R分量
    merged = cv2.addWeighted(img_gray, 1, img_rgb, 1, 0)
    cv2.imwrite(str(clusters_number)+'cluster_merged_'+str(k)+'.png', merged)
    
    for l in range(length):
        for m in range(width):
            for n in range(3):
                if img[l][m][n] == 0:
                    img[l][m][n] = img_rgb[l][m][n]
                    
cv2.imwrite(str(clusters_number)+'cluster_merged_.png', img)
y_arr = y_arr -1 #跑完後再減掉，以免需要再跑一次時跑出錯

#%% prominence-plot
os.chdir(path+'/wholeimg')#指定路徑

#各別畫
for i in range(cluster_avg.shape[1]):
    
    #Mean
    #plt.subplot(211)
    m = cluster_avg[:,i]
    peaks, _ = find_peaks(m, prominence=0.015)
    
    plt.rcParams['figure.figsize'] = (8.0, 6.0) # 設置figure_size
    plt.rcParams['image.interpolation'] = 'nearest' # 設置 interpolation style
    plt.rcParams['savefig.dpi'] = 100 #圖片像素
    plt.rcParams['figure.dpi'] = 100 #分辨率
    
    plt.plot(wavelength, m, label='cluster'+str(i+1), color=colormap[i])
    plt.plot(wavelength[peaks], m[peaks], "ob"); plt.legend(['prominence'])
    for j in range(peaks.shape[0]):
        plt.annotate(str(wavelength[peaks[j]]), (wavelength[peaks[j]], m[peaks[j]])) #標記峰值
       
    plt.legend(loc='right',bbox_to_anchor=(1.23,0.5))
    plt.title(cellname+'_cluster'+str(i+1)+'_mean')
    plt.xlabel("wavelength")
    plt.ylabel("Transimittance")
    x_ticks = np.arange(400,1000,50)
    y_ticks = np.arange(0,2,0.1)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.tight_layout()
    plt.savefig('cluster'+str(i+1)+'_avg.png')
    plt.clf()
    
    
    #standard deviation
    #plt.subplot(212)
    t = cluster_std[:,i]
    
    plt.rcParams['figure.figsize'] = (8.0, 6.0) # 設置figure_size
    plt.rcParams['image.interpolation'] = 'nearest' # 設置 interpolation style
    plt.rcParams['savefig.dpi'] = 100 #圖片像素
    plt.rcParams['figure.dpi'] = 100 #分辨率
    
    plt.plot(wavelength, t, label='cluster'+str(i+1), color=colormap[i])
       
    plt.legend(loc='right',bbox_to_anchor=(1.23,0.5))
    plt.title(cellname+'_cluster'+str(i+1)+'_std')
    plt.xlabel("wavelength")
    plt.ylabel("Transimittance")
    x_ticks = np.arange(400,1000,50)
    y_ticks = np.arange(0,2,0.1)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.tight_layout()
    plt.savefig('cluster'+str(i+1)+'_std.png')
    #plt.show    
    plt.clf()
    
    #Error bar
    
    plt.rcParams['figure.figsize'] = (8.0, 6.0) # 設置figure_size
    plt.rcParams['image.interpolation'] = 'nearest' # 設置 interpolation style
    plt.rcParams['savefig.dpi'] = 100 #圖片像素
    plt.rcParams['figure.dpi'] = 100 #分辨率
 
    plt.errorbar(wavelength, m, t, ecolor=colormap[i], lw=1, alpha = 0.6)
    plt.errorbar(wavelength, m, t+t, ecolor=colormap[i], lw=1, alpha = 0.3)
    
    plt.plot(wavelength, m, label='cluster'+str(i+1), color=colormap[i])
    
    plt.legend(loc='right',bbox_to_anchor=(1.23,0.5))
    plt.title(cellname+'_cluster'+str(i+1)+'_errorbar')
    plt.xlabel("wavelength")
    plt.ylabel("Transimittance")
    x_ticks = np.arange(400,1000,50)
    y_ticks = np.arange(0,2,0.1)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.tight_layout()
    plt.savefig('cluster'+str(i+1)+'_errorbar.png')
  
    plt.clf()
    
    

#%%
#    #Table
#    plt.subplot(131)
#    row_labels = ['Pixel number','Mean','Standard Deviation','Maximum','Minimum']
#    data = [[cluster_point.shape[0]], [np.mean(t)], [np.mean(m)],[np.max(t)], [np.min(t)] ] 
#    plt.table(cellText=data,rowLabels=row_labels,loc="center")
    
#%%
#cell-blank-edge

cbe1 = ['cell','blank','edge']
cbe2 = [[3,4,6,7,8],
        [1,2,5,9], 
        []]
    
plt.rcParams['figure.figsize'] = (8.0, 6.0) # 設置figure_size
plt.rcParams['image.interpolation'] = 'nearest' # 設置 interpolation style
plt.rcParams['savefig.dpi'] = 100 #圖片像素 
plt.rcParams['figure.dpi'] = 100 #分辨率

for m in range(len(cbe1)):
    
    #plt.subplot(121)
    for i in range(cluster_avg.shape[1]):   
        t = cluster_avg[:,i]
        peaks, _ = find_peaks(t, prominence=0.015)       
        for k in range(len(cbe2[m])):
            if i == cbe2[m][k]-1:            
                plt.plot(wavelength, t, label='cluster'+str(i+1), color=colormap[i])
                plt.plot(wavelength[peaks], t[peaks], "ob"); plt.legend(['prominence'])
                for j in range(peaks.shape[0]):
                    plt.annotate(str(wavelength[peaks[j]]), (wavelength[peaks[j]], t[peaks[j]])) #標記峰值 
           
        plt.legend(loc='right',bbox_to_anchor=(1.23,0.5))
        plt.title(cellname+'_'+cbe1[m]+'_mean')
        plt.xlabel("wavelength")
        plt.ylabel("Transimittance")
        x_ticks = np.arange(400,1000,50)
        y_ticks = np.arange(0,2,0.1)
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
    plt.tight_layout()
    plt.savefig('cluster_'+cbe1[m]+'_avg.png')
    plt.clf()
    
    
    #plt.subplot(122)
    for i in range(cluster_std.shape[1]):   
        t = cluster_std[:,i]   
        for k in range(len(cbe2[m])):
            if i == cbe2[m][k]-1:            
                plt.plot(wavelength, t, label='cluster'+str(i+1), color=colormap[i])

        plt.legend(loc='right',bbox_to_anchor=(1.23,0.5))
        plt.title(cellname+'_'+cbe1[m]+'_std')
        plt.xlabel("wavelength")
        plt.ylabel("Transimittance")
        x_ticks = np.arange(400,1000,50)
        y_ticks = np.arange(0,2,0.1)
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
    plt.tight_layout()
    plt.savefig('cluster_'+cbe1[m]+'_std.png')
    plt.clf()

#%% 基於常態分佈法則，試找出超過兩個標準差外的值

    