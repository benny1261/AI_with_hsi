import os
from spectral.io import envi
import spectral
import cv2
import numpy as np
import glob

OUTPUT_CHANNEL = 50
DEFECTIVE_THRESHOLD = 10
EXPORT_RAW = True

def delete_defective(hsi):
    '''eliminate super high value caused by missing data\n
    ret -> fixed image'''
    dpi = np.where(hsi > DEFECTIVE_THRESHOLD)       # defective pixel indices
    coordinates = list(zip(dpi[0], dpi[1], dpi[2]))
    fixed_value = []

    for _ in coordinates:
        row, col, dep = _[0], _[1], _[2]
        if row == 0 & col == 0:
            patch = hsi[0:row+2, 0:col+2, dep]
        elif row == 0:
            patch = hsi[0:row+2, col-1:col+2, dep]
        elif col == 0:
            patch = hsi[row-1:row+2, 0:col+2, dep]
        else:
            patch = hsi[row-1:row+2, col-1:col+2, dep]

        masked = np.ma.masked_where(patch > DEFECTIVE_THRESHOLD, patch)
        fixed_value.append(np.mean(masked))

    # seperate into two loops to prevent taking mean after value replace
    for i in range(len(coordinates)):
        row, col, dep = coordinates[i][0], coordinates[i][1], coordinates[i][2]
        hsi[row, col, dep] = fixed_value[i]
    return hsi

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
        export_img = arr

        if EXPORT_RAW:
            arrc = delete_defective(arr.copy())
            envi.save_image(name+'_fixed.hdr', arrc, interleave = "bsq", ext = "raw")
            # use original hdr file to enable imec software reading
            os.system(f'copy {name}.hdr {name}_fixed.hdr')
            export_img = arrc

        img_gray = cv2.normalize(export_img[:,:,OUTPUT_CHANNEL], None,0,255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imwrite(name+'.jpg', img_gray)