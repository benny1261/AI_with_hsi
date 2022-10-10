# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
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
from sklearn.decomposition import IncrementalPCA
#from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

#%%

import random

os.chdir(path)
scp = np.loadtxt(open("scp.csv"),delimiter=",",skiprows=0, dtype='int') #single cell position

randomlist = random.sample(range(0, scp.shape[0]-1), 9)
scp = scp[[randomlist],:]
scp = scp.reshape(len(randomlist),4)

cell = cbe2[0]
blank = cbe2[1]
edge = cbe2[2]

target_cluster = cbe2[0] #先跑whole_img看完圖後，知道細胞是屬於哪個cluster 
#tc = [] #用來搜集上述資訊

if os.path.exists(path+'/singlecell'):
    print ("Folder exist")
else:
    os.mkdir(path+'/singlecell')#建立資料夾
    print ("Folder is created")
    

#%%


#%%----------------------------------------------------------------------2mins

#os.chdir(path)#讀取資料夾

#spectral.settings.envi_support_nonlowercase_params = 'TRUE'
#
#img = envi.open("cube_envi32.hdr" , "cube_envi32.dat")
#
#
#print(img.__class__)
#print(img)
#print('===============================')
#
#arr = img.load()
#arr.__class__
#print(arr.info())
#print("Shape:")
#print(arr.shape)
#
#metadata =img.metadata
#arr = arr[:,:,0:height] #只取400-700mm
#
##data = arr[:,:,1]
##data = data.reshape(1216,1936)
##
##data = cv2.normalize(data, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC3)



#%%----------flatten-----------------------------------------------------15min
#new_arr = np.zeros((length*width,height), dtype='float16', order='C')
#for i in range(length):
#    for j in range(width):
#        new_arr[width * i + j, :] = arr[i,j, :]
#        
##----------IPCA降維---------------------
#
#ipca = IncrementalPCA(n_components=n_components, batch_size=10)
#newarr_ipca = ipca.fit_transform(new_arr)
#
#
##----------kmeans分群----------------------------------------------------------
#
#kmeans = KMeans(n_clusters = clusters_number)
#kmeans.fit(newarr_ipca)
#y_kmeans = kmeans.predict(newarr_ipca)
#
##-----------回復成xy------------------------------------------------------------
#y_arr = np.zeros((length,width),dtype='int')
#for i in range(length*width):
#    a = i//width; #兩槓取整數 一槓有小數
#    b = i%width
#    y_arr[a,b] = y_kmeans[i]
#
#np.savetxt(str(clusters_number)+'_cluster_result.csv', y_arr, fmt="%d", delimiter=",")
#
#cluster_result = cv2.normalize(y_arr, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC3)
#
#cv2.imwrite(str(clusters_number)+'_cluster_result.png', cluster_result)

#%%--------分別取出每個cluster的位置，存成新的矩陣----------------------------25mins

os.chdir(path+'/singlecell')#讀取資料夾

for l in range(len(target_cluster)):
    
    tc = []
    for cells in range(scp.shape[0]):
        
        cluster_points = []
        for k in range(clusters_number):
            cluster_point = []
            #result = np.zeros((1216,1936),dtype='int')
            for i in range(scp[cells,1],scp[cells,3]):
                for j in range(scp[cells,0],scp[cells,2]):
                    if y_arr[i][j] == k :
                        #result[i][j] = k
                        cluster_point.append([i,j])
            cluster_point = np.asarray(cluster_point)
            cluster_points.append(cluster_point)
        cluster_points = np.asarray(cluster_points)
        
        #---------將位置帶回3d高光譜-並依cluster計算平均值，畫出curve------------------------
        avg_cluster = []
        for k in range(clusters_number):
            cluster_point = cluster_points[k]
            avg_wave = []
            for z in range(height): 
                total = 0
                for point in range(cluster_point.shape[0]):
                    total = total + arr[cluster_point[point][0],cluster_point[point][1],z]
                
                if cluster_point.shape[0] == 0:
                    avg = total / 10000
                else:
                    avg = total / cluster_point.shape[0]  
                
                avg_wave.append(avg)
            avg_wave = np.asarray(avg_wave)
            avg_cluster.append(avg_wave)
        avg_cluster = np.asarray(avg_cluster)
        
        #-------很重要！----搜集同個cluster的資訊 for single cell---------------------
        tc.append(avg_cluster[target_cluster[l]-1,:])
                
        np.savetxt('cell'+str(cells+1)+'_'+str(clusters_number)+'cluster_avg.csv', avg_cluster, fmt="%f", delimiter=",")
            
        #--------將每個singlecell的16群光譜plot出來----------------------------------
        plt.rcParams['figure.figsize'] = (8.0, 6.0) # 設置figure_size
        plt.rcParams['image.interpolation'] = 'nearest' # 設置 interpolation style
        plt.rcParams['savefig.dpi'] = 100 #圖片像素
        plt.rcParams['figure.dpi'] = 100 #分辨率
        #存16合1
        for i in range(avg_cluster.shape[0]):
            
            if l != 0:
                break
            
            plt.plot(wavelength,avg_cluster[i,:],label = 'cluster'+str(i+1),color=colormap[i])
            plt.legend(loc='right',bbox_to_anchor=(1.23,0.5))
            plt.title(cellname +'_cell' + str(cells+1))
            plt.xlabel("wavelength")
            plt.ylabel("Transimittance")
            x_ticks = np.arange(400,1000,50)
            y_ticks = np.arange(0,2,0.1)
            plt.xticks(x_ticks)
            plt.yticks(y_ticks)
            
        if l == 0:
            plt.tight_layout()
            plt.savefig('cell'+str(cells+1)+'_'+str(clusters_number)+'cluster_all_avg.png')
            plt.clf()
#        #16張各別存
#        for i in range(avg_cluster.shape[0]):
#            
#            if l != 0:
#                break
#            
#            plt.plot(wavelength,avg_cluster[i,:],label = 'cluster'+str(i+1),color=colormap[i])
#            plt.legend(loc='right',bbox_to_anchor=(1.23,0.5))
#            plt.title(cellname + '_cell' + str(cell+1))
#            plt.xlabel("wavelength")
#            plt.ylabel("Transimittance")
#            x_ticks = np.arange(400,725,50)
#            y_ticks = np.arange(0,1.5,0.1)
#            plt.xticks(x_ticks)
#            plt.yticks(y_ticks)
#            plt.tight_layout()
#            plt.savefig('cell'+str(cell+1)+'_'+str(clusters_number)+'cluster'+str(i+1)+'_avg.png')
#            plt.clf()
          
        #cell
        for i in range(avg_cluster.shape[0]):
            if l > len(target_cluster):
                break
            for j in range(len(cell)):
                if i == cell[j]-1:            
                    plt.plot(wavelength,avg_cluster[i,:],label = 'cluster'+str(i+1),color=colormap[i])
                    plt.legend(loc='right',bbox_to_anchor=(1.23,0.5))        
            plt.title(cellname+'_cell')
            plt.xlabel("wavelength")
            plt.ylabel("Transimittance")
            x_ticks = np.arange(400,1000,50)
            y_ticks = np.arange(0,2,0.1)
            plt.xticks(x_ticks)
            plt.yticks(y_ticks)
        
        plt.tight_layout()
        plt.savefig('cell'+str(cells+1)+'_'+str(clusters_number)+'cluster_cell_avg.png')
        plt.clf()
        
        #blank
        for i in range(avg_cluster.shape[0]):
            if l > len(target_cluster):
                break
            for j in range(len(blank)):
                if i == blank[j]-1:            
                    plt.plot(wavelength,avg_cluster[i,:],label = 'cluster'+str(i+1),color=colormap[i])
                    plt.legend(loc='right',bbox_to_anchor=(1.23,0.5))        
            plt.title(cellname+'_blank')
            plt.xlabel("wavelength")
            plt.ylabel("Transimittance")
            x_ticks = np.arange(400,1000,50)
            y_ticks = np.arange(0,2,0.1)
            plt.xticks(x_ticks)
            plt.yticks(y_ticks)
        
        plt.tight_layout()
        plt.savefig('cell'+str(cells+1)+'_'+str(clusters_number)+'cluster_blank_avg.png')
        plt.clf()
        
        #edge
        for i in range(avg_cluster.shape[0]):
            if l > len(target_cluster):
                break
            for j in range(len(edge)):
                if i == edge[j]-1:            
                    plt.plot(wavelength,avg_cluster[i,:],label = 'cluster'+str(i+1),color=colormap[i])
                    plt.legend(loc='right',bbox_to_anchor=(1.23,0.5))        
            plt.title(cellname+'_edge')
            plt.xlabel("wavelength")
            plt.ylabel("Transimittance")
            x_ticks = np.arange(400,1000,50)
            y_ticks = np.arange(0,2,0.1)
            plt.xticks(x_ticks)
            plt.yticks(y_ticks)
        
        plt.tight_layout()
        plt.savefig('cell'+str(cells+1)+'_'+str(clusters_number)+'cluster_edge_avg.png')
        plt.clf()
            
    #----
    tc = np.asarray(tc)
    tc_avg = np.mean(tc, axis = 0)
    tc_std = np.std(tc, axis = 0)
    
    np.savetxt('singlecell_cluster'+str(target_cluster[l])+'.csv', tc, fmt="%f", delimiter=",")
    
    plt.rcParams['figure.figsize'] = (8.0, 6.0) # 設置figure_size
    plt.rcParams['image.interpolation'] = 'nearest' # 設置 interpolation style
    plt.rcParams['savefig.dpi'] = 100 #圖片像素
    plt.rcParams['figure.dpi'] = 100 #分辨率
    for i in range(tc.shape[0]):
        #plt.plot(wavelength,tc[i,:] ,label = 'cell'+str(i+1)+'_cluster'+str(target_cluster[l]),color=colormap[i])
        plt.plot(wavelength,tc[i,:] ,label = 'cell'+str(i+1), color=colormap[i])
        plt.legend(loc='right',bbox_to_anchor=(1.23,0.5))
        plt.title(cellname + '_cluster' + str(target_cluster[l]))
        plt.xlabel("wavelength")
        plt.ylabel("Transimittance")
        x_ticks = np.arange(400,1000,50)
        y_ticks = np.arange(0,2,0.1)
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
    plt.tight_layout()
    plt.savefig('singlecell_cluster'+str(target_cluster[l])+'.png')
    plt.clf()

    # (number) cells avg
    plt.rcParams['figure.figsize'] = (8.0, 6.0) # 設置figure_size
    plt.rcParams['image.interpolation'] = 'nearest' # 設置 interpolation style
    plt.rcParams['savefig.dpi'] = 100 #圖片像素
    plt.rcParams['figure.dpi'] = 100 #分辨率
    plt.plot(wavelength, tc_avg, label = str(len(randomlist))+'cellavg', color='black')
    plt.legend(loc='right', bbox_to_anchor=(1.23,0.5))
    plt.title(cellname + '_cluster' + str(target_cluster[l])+'_mean')
    plt.xlabel("wavelength")
    plt.ylabel("Transimittance")
    x_ticks = np.arange(400,1000,50)
    y_ticks = np.arange(0,2,0.1)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.tight_layout()
    plt.savefig(str(len(randomlist))+'cell_cluster'+str(target_cluster[l])+'_avg.png')
    plt.clf()

    # (number) cells std
    plt.rcParams['figure.figsize'] = (8.0, 6.0) # 設置figure_size
    plt.rcParams['image.interpolation'] = 'nearest' # 設置 interpolation style
    plt.rcParams['savefig.dpi'] = 100 #圖片像素
    plt.rcParams['figure.dpi'] = 100 #分辨率
    plt.plot(wavelength, tc_std, label = str(len(randomlist))+'cellavg', color='black')
    plt.legend(loc='right', bbox_to_anchor=(1.23,0.5))
    plt.title(cellname + '_cluster' + str(target_cluster[l])+'_std')
    plt.xlabel("wavelength")
    plt.ylabel("Transimittance")
    x_ticks = np.arange(400,1000,50)
    y_ticks = np.arange(0,2,0.1)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.tight_layout()
    plt.savefig(str(len(randomlist))+'cell_cluster'+str(target_cluster[l])+'_std.png')
    plt.clf()


#%%------存下每個cluster對應位置，套色疊圖--------------------------------------2mins

os.chdir(path+'/singlecell')#指定路徑

            
img_br = np.mat(np.zeros((length,width)),dtype=np.uint8)  

y_arr = y_arr + 1 #把所有元素都加1，避免0出現，帶進去全零矩陣時看不見

for cells in range(scp.shape[0]):
    for k in range(1,1+clusters_number):
        result = np.zeros((length,width),dtype='int')
        for i in range(scp[cells,1],scp[cells,3]):
            for j in range(scp[cells,0],scp[cells,2]):
                if y_arr[i][j] == k:
                    result[i][j] = y_arr[i][j]
                    
        result = result/(k)
        img_r = result*cm[k-1][0]
        img_g = result*cm[k-1][1]
        img_b = result*cm[k-1][2]
    
        img_rgb = cv2.merge([img_b,img_g,img_r]).astype(np.uint8)#合併B、G、R分量
        merged = cv2.addWeighted(img_gray, 1, img_rgb, 1, 0)        
        merged = merged[scp[cells,1]:scp[cells,3],scp[cells,0]:scp[cells,2],:]
        
        cv2.imwrite('cell'+str(cells+1)+'_'+str(clusters_number)+'cluster_merged_'+str(k)+'.png', merged)
    
    
y_arr = y_arr -1 #跑完後再減掉，以免需要再跑一次時出錯

#%%

