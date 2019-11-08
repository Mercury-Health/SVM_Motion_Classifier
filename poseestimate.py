# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:25:49 2019

@author: Michael K
"""

import cv2

BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
"LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
"RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
"Background": 15}

def get_model():
    OPENPOSE_DIR="C:/Users/Michael K/Desktop/Mercury_Health/SegCNN/openpose-master/models/"
    POSE_FOLDER=OPENPOSE_DIR+"pose/body_25/"
    PROTO=POSE_FOLDER+"pose_deploy.prototxt"
    MPI_MODEL=POSE_FOLDER+"pose_iter_584000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(PROTO,MPI_MODEL)
    return net

def poseestimate(net,frame,thr,frameWidth, frameHeight):
    inp = cv2.dnn.blobFromImage(frame, 1.0 / 255, (frameWidth, frameHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0,i,:,:]
        
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thr else None)
    return points