'''
this is used to analyze specified csv file exported from spectrum_observer
'''
import csv
import os
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import numpy as np

def read_csv_to_dict(file_path, sample:int= None)-> Dict[str, float]:
    global wavelengths
    result_dict = {}
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        wavelengths = list(map(float, header[1:]))
        for row in csvreader:
            key = row[0]
            values = list(map(float, row[1:]))  # convert elements to float
            result_dict[key] = values
    if sample is not None:
        result_dict = sample_dict(result_dict, sample)
    return result_dict

def sample_dict(orig_dict:dict, k:int)-> dict:
    return dict(random.sample(list(orig_dict.items()), k))

def show_ccmatrix(*data:dict):
    lengths= []
    dataflatten= []
    for set in data:
        lengths.append(len(set))
        dataflatten.extend(set.values())

    correlation_matrix = np.corrcoef(np.asarray(dataflatten))

    if len(lengths) == 1:
        plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()

    else:
        fig, axes = plt.subplots(len(lengths),len(lengths), gridspec_kw={'width_ratios': lengths, 'height_ratios': lengths})
        for ind_row in range(len(lengths)):
            for ind_col in range (len(lengths)):
                if ind_col > ind_row:
                    axes[ind_row,ind_col].axis('off')
                else:
                    cm_slice = correlation_matrix[sum(lengths[:ind_row]):sum(lengths[:ind_row+1]),
                                                sum(lengths[:ind_col]):sum(lengths[:ind_col+1])]
                    cm_slice_mean = np.mean(cm_slice)
                    axes[ind_row,ind_col].set_title(str(cm_slice_mean), fontsize= 'small')
                    if ind_col==0 and ind_row==0:
                        ax00 = axes[0,0].imshow(cm_slice, cmap='coolwarm', vmin=-1, vmax=1)
                    else:
                        axes[ind_row,ind_col].imshow(cm_slice, cmap='coolwarm', vmin=-1, vmax=1)

        fig.colorbar(ax00, ax=axes[0,len(lengths)-1])

    plt.suptitle("Correlation Coefficient Matrix")
    plt.tight_layout()

def dict2array(data:dict, meanstd:bool= False)->np.ndarray:
    _list = []
    for line in data.values():
        _list.append(line)
    block =  np.array(_list)
    if meanstd:
        means = np.mean(block, axis= 0)
        stds = np.std(block, axis= 0)
        return block, means, stds
    return block

if __name__ == '__main__':

    PATH:str = r'C:\Users\user\Desktop\analysis'

    ctcs = read_csv_to_dict(os.path.join(PATH, 'ctc.csv'), sample= 20)
    wbc_20230617_v10_3 = read_csv_to_dict(os.path.join(PATH, 'wbc_20230617_v10-3.csv'), sample= 20)
    wbc_20230703_v1_2 = read_csv_to_dict(os.path.join(PATH, 'wbc_20230703_v1-2.csv'), sample= 20)
    wbc_20230707_v4_8 = read_csv_to_dict(os.path.join(PATH, 'wbc_20230707_v4-8.csv'), sample= 20)

    # correlation coefficient --------------------------------------------------------------
    show_ccmatrix(ctcs, wbc_20230617_v10_3, wbc_20230703_v1_2, wbc_20230707_v4_8)

    # Plot mean and std --------------------------------------------------------------------
    # ctc_arr, ctc_mean, ctc_std = dict2array(ctcs, True)
    # wbc_20230617_v10_3_arr, wbc_20230617_v10_3_mean, wbc_20230617_v10_3_std = dict2array(wbc_20230617_v10_3, True)
    # wbc_20230703_v1_2_arr, wbc_20230703_v1_2_mean, wbc_20230703_v1_2_std = dict2array(wbc_20230703_v1_2, True)
    # wbc_20230707_v4_8_arr, wbc_20230707_v4_8_mean, wbc_20230707_v4_8_std = dict2array(wbc_20230707_v4_8, True)

    # plt.plot(wavelengths, ctc_mean, label='ctc', color= 'red')
    # plt.fill_between(wavelengths, ctc_mean - ctc_std, ctc_mean + ctc_std, alpha=0.3, color= 'red')
    # plt.plot(wavelengths, wbc_20230617_v10_3_mean, label='wbc_20230617_v10_3', color= 'blue')
    # plt.fill_between(wavelengths, wbc_20230617_v10_3_mean - wbc_20230617_v10_3_std, wbc_20230617_v10_3_mean + wbc_20230617_v10_3_std, alpha=0.3, color= 'blue')
    # plt.plot(wavelengths, wbc_20230703_v1_2_mean, label='wbc_20230703_v1_2', color= 'green')
    # plt.fill_between(wavelengths, wbc_20230703_v1_2_mean - wbc_20230703_v1_2_std, wbc_20230703_v1_2_mean + wbc_20230703_v1_2_std, alpha=0.3, color= 'green')
    # plt.plot(wavelengths, wbc_20230707_v4_8_mean, label='wbc_20230707_v4_8', color= 'brown')
    # plt.fill_between(wavelengths, wbc_20230707_v4_8_mean - wbc_20230707_v4_8_std, wbc_20230707_v4_8_mean + wbc_20230707_v4_8_std, alpha=0.3, color= 'brown')

    # plt.xlabel('Wavelengths')
    # plt.ylabel('Transmittance')
    # plt.title('Average Line Plot with Standard Deviation')
    # plt.legend()
    # plt.grid(True)

# ----------------------------------------------------------
    # Adjusting layout to prevent overlapping of titles and labels
    # plt.tight_layout()

    # Show the plot
    plt.show()