#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 01:03:02 2021

@author: tc_chien
"""

import os, sys, random
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from shutil import copyfile

#%%
    
path = '/Volumes/T7 Touch/HSI-dataset/00-Analysis/07-gray-avg'

annotations = glob(path + '/*.xml')

nl = ['l5', 'l6', 'l7', 'w5', 'w6', 'w7', 'o']

df = []

for n in nl:
    
    for file in annotations:

        filename = file.split('/')[-1]
        filename =filename.split('.')[0] + '-' + n + '.png'
        row = []
        parsedXML = ET.parse(file)
        for node in parsedXML.getroot().iter('object'):
            blood_cells = node.find('name').text
            xmin = int(node.find('bndbox/xmin').text)
            xmax = int(node.find('bndbox/xmax').text)
            ymin = int(node.find('bndbox/ymin').text)
            ymax = int(node.find('bndbox/ymax').text)
            
            if n == 'l5':
                xmin = round(xmin*0.625)
                xmax = round(xmax*0.625)
            elif n == 'l6':
                xmin = round(xmin*0.75)
                xmax = round(xmax*0.75)
            elif n == 'l7':
                xmin = round(xmin*0.875)
                xmax = round(xmax*0.875)
            elif n == 'w5':
                xmin = round(ymin*0.625)
                xmax = round(ymax*0.625)
            elif n == 'w6':
                xmin = round(ymin*0.75)
                xmax = round(ymax*0.75)
            elif n == 'w7':
                xmin = round(ymin*0.875)
                xmax = round(ymax*0.875)
            elif n == 'o':
                pass
            
            row = [filename, blood_cells, xmin, xmax, ymin, ymax]
            
            df.append(row)

data = pd.DataFrame(df, columns=['filename', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax'])

os.chdir(path)

data[['filename', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv('test.csv', index=False)
data[['xmin', 'ymin', 'xmax', 'ymax']].to_csv('scp.csv', header=None, index=False)

#%%

