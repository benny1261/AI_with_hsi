import os
from spectral.io import envi
import spectral
import cv2
import numpy as np
import glob

OUTPUT_CHANNEL = 20

path = os.path.dirname(__file__)+r'/data'
os.chdir(path)
spectral.settings.envi_support_nonlowercase_params = 'TRUE'

hdr_list, raw_list = glob.glob('*.hdr'), glob.glob('*.raw')
hsi_dict = {}
if (not hdr_list) or (not raw_list):
    raise FileNotFoundError
elif len(hdr_list) != len(raw_list):
    raise IOError("number of .hdr .raw are not equal")
else:
    for _ in hdr_list:
        name = _.split(".")[0]
        hsi_dict[name] = envi.open(name+ ".hdr" , name+ ".raw")     # our hsi metadata stored in ENVI raster format
        arr = hsi_dict[name].load()
        print(name,"-----------------")
        print(arr.info())
        print("shape=", arr.shape)
        img_gray = cv2.normalize(np.float32(arr[:,:,OUTPUT_CHANNEL]), None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC3)
        cv2.imwrite(name+'.jpg', img_gray)