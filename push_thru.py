# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:23:59 2019

@author: Michael K
"""

import cv2
import poseestimate
from motiontrack import motiontrack
import numpy as np

def pend(vec):
    flag = 1
    for i in range(len(vec)):
        if flag == 1 and vec[int(i)] == -1:
            ind = np.argmax(np.asarray(vec) > 0)
            vec[int(i)] = vec[ind]
        elif flag == 1 and vec[int(i)] != -1:
            flag = 0
            storeval = vec[int(i)]
        elif flag == 0 and vec[int(i)] != -1:
            storeval = vec[int(i)]
        else:
            vec[int(i)] = storeval      
    return vec
            
        

def forward_pass(net,file_path,min_area):
    #read frames
    vid = cv2.VideoCapture(file_path)
    success,image = vid.read()
    success = True
    frames = [];
    frames.append(image)
    while success:
        success,image = vid.read()
        frames.append(image)
    
    frames = frames[0:-2]
    stop = len(frames) - 1
    #create position vector and get pose data.
    pos_x = []
    pos_y = []
    poses = []
    for i in list(range(stop)):
        [x,y,w,h,flag] = motiontrack(frames[i+1],frames[i],min_area)
        if flag:
            #keep previous xywh
            pos_x.append(-1)
            pos_y.append(-1)
        else:  
            pos_x.append(x+w)
            pos_y.append(y+h)
        #poses.append([poseestimate.poseestimate(net,frames[i+1],0.5,256,256)])
    return [pend(pos_x), pend(pos_y), pend(poses)]