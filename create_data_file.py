# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:51:31 2019

@author: Michael K
"""

import poseestimate
from push_thru import forward_pass
import os
import numpy as np

#file_path = "C:/Users/Michael K/Desktop/real/Falling_all/0a75922e-bd3a-11e9-8ade-60f81da93ee0cam1.avi"
base_dir = "C:/Users/Michael K/Desktop/real/"
keyword = "Rolling_Bed_Val"
top_dir = base_dir + keyword
net = poseestimate.get_model() #probably put hyperparameters in here
toSave = []
min_area = 0.5

i = 1
for file in os.listdir(top_dir):
    if 'joints' in file:
        continue
    file_path = top_dir + '/' + file
    [pos_x, pos_y, poses] = forward_pass(net, file_path,min_area)
    pos_x = np.asarray(pos_x)
    pos_y = np.asarray(pos_y)
    poses = np.asarray(poses)
    label = keyword
    
    #toSave.append([pos_x,pos_y,poses,label])
    if len(pos_x) > 0:
        if min(pos_x) > 0:
            toSave.append([pos_x,pos_y,label])
            print(i)
    i += 1
    
np.save(base_dir+keyword,toSave)