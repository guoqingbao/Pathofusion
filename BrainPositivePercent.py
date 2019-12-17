# Author: Guoqing Bao
# School of Computer Science, The University of Sydney
# Date: 2019-12-12
# GitHub Project Link: https://github.com/guoqingbao/Pathofusion
# Please cite our work if you found it is useful for your research or clinical practice


# from IPython import get_ipython

# %%

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
from PIL import Image
# enable to process extremely large images
Image.MAX_IMAGE_PIXELS = 10000000000
import cv2
import sqlite3
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import pandas as pd
import cv2
from urllib.request import urlopen
import io
import gc
gc.enable()


# %%
project_path = "./"
path = project_path + "results/prediction/"
PATIENT_ID = 23


# %%
prediction_array = np.load(path + 'prediction_array'+str(PATIENT_ID) + '.npy')
intensity_array = np.load(path + 'intensity_array'+str(PATIENT_ID) + '.npy')


# %%
d_intensity = np.array(intensity_array)
bk_position = np.where(d_intensity>230)[0].tolist()

# Calculate the percentage of immunopositive in each feature
# %%
fetures = ['Necrosis-palisading', 'MicVas-Proliferation', 'Blood-Vessel', 'Necrosis-Geo', 'Brain-Tissue', 'Tumor']

labels = ["CD276 Positive Alone",
          "Feature with positive",
          "Feature without positive"]   
colors_merged = []

# colors_merged.append([1,0,0]) #, positive alone
colors_merged.append([1,0,1]) # intersection --> (magenta)
colors_merged.append([0,0,1]) #, feature alone 
    
for k in range(6):
    print("\r\nDetecting percentage of immunopositive area for feature: ", fetures[k])
    colors = []
    for i in range(6):
        colors.append([0,0,0])
        
    colors[k] = [0,0,1] # detect feature k
    
    typeret = np.argmax(prediction_array,axis=1)
    typeret = typeret + 1
    typeret[bk_position]=0
    totalArea = np.sum(typeret!=0)
    colors = np.array(colors)
    
    predicts = []
    predicts_bin = []

    for item in prediction_array:
        predicts.append(colors[np.argmax(item)] * np.max(item))
        predicts_bin.append(colors[np.argmax(item)] * 1.0)

    predicts = np.array(predicts)    
    predicts_bin = np.array(predicts_bin)   
    
    r_channel = predicts_bin[:,0] 
    r_channel[bk_position] = 0
    g_channel = predicts_bin[:,1] 
    g_channel[bk_position] = 0
    b_channel = predicts_bin[:,2] 
    b_channel[bk_position] = 0
    r = r_channel.reshape(rows,cols)
    g = g_channel.reshape(rows,cols)
    b = b_channel.reshape(rows,cols)

    heatmap_bin = cv2.merge([r, g,b])
    
    im1 =  (heatmap_bin * 255).astype(np.uint8)
    im2_aligned_excluded = cv2.imread(path + 'brain_cd276_heatmap_aligned_bin' + str(PATIENT_ID) + '.bmp')
    positive_mask = np.all(im2_aligned_excluded == [0, 0, 255], axis=-1)
    positive_mask = (positive_mask*255).astype(np.uint8)

    im2_aligned_excluded1 = cv2.cvtColor(im2_aligned_excluded, cv2.COLOR_BGR2RGB)
    im2_aligned_excluded1 = cv2.bitwise_and(im2_aligned_excluded1, im2_aligned_excluded1, mask=positive_mask)

    combined = im1 | im2_aligned_excluded1
    # bgrCombined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(im2_aligned_excluded,cv2.COLOR_BGR2GRAY)
    _,mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY);

    bgrCombined_excluded = cv2.bitwise_and(combined, combined, mask=mask)
#     plt.imshow(bgrCombined_excluded)
   
   
    
    gray = cv2.cvtColor(bgrCombined_excluded,cv2.COLOR_BGR2GRAY)
    _,allMask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY);
    total = np.sum(allMask==255) #total pixels excludes picture black background
    
    counts1 = []
    rets = []
    fig, ax = plt.subplots(1,2, sharex=True, figsize=(20,12))
    index = 0
    percents = []
    for color in colors_merged:    
        c = np.uint8([[color]])
        hsv_c = cv2.cvtColor(c,cv2.COLOR_BGR2HSV)
        hsv = cv2.cvtColor(bgrCombined_excluded, cv2.COLOR_BGR2HSV)
        hue_low = hsv_c[0][0][0]-5
        if hue_low <0:
            hue_low = 0
        hue_high = hsv_c[0][0][0]+5
        if color == [1,1,1]:   
            lower = np.array([0,0,100])
            upper = np.array([255,50,255])
        elif color == [1,0.7,0.7]:
            lower = np.array([hue_low,77-20,100])
            upper = np.array([hue_high,77+30,255])    

        else:
            lower = np.array([hue_low,90,130])
            upper = np.array([hue_high,hsv_c[0][0][1],255])

        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(bgrCombined_excluded,bgrCombined_excluded, mask= mask)
        rets.append(res)
        counts1.append(np.sum(mask==255))
#         print("HSV color range:", lower, upper, " account {0:.2f}%".format(np.sum(mask==255)/total*100))
        ax[index].imshow(res)
        ax[index].title.set_text(labels[index] +" (Area={0:.2f}%)".format(np.sum(mask==255)/total*100))
        index += 1
        percents.append(np.sum(mask==255)/total*100)
    plt.show() 
    
    print(fetures[k], ": Percentage of immunopositive area is ", percents[0]/(percents[0]+percents[1]))


# %%



