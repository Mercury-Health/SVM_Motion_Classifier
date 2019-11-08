# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:12:05 2019

@author: Michael K
"""
import imutils
import cv2
import numpy as np

def motiontrack(frameCur, framePrev, min_area):
    # resize the frame, convert it to grayscale, and blur it
    frame0 = imutils.resize(frameCur, width=500)
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray0 = cv2.GaussianBlur(gray0, (21, 21), 0)
    
    frame1 = imutils.resize(framePrev, width=500)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
    
    frameDelta = cv2.absdiff(gray0,gray1)
    #rameDelta = cv2.cvtColor(frameDelta, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(frameDelta,25,255, cv2.THRESH_BINARY)[1]

    
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # loop over the contours
    areas = []
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            areas.append(0)
        else:
            areas.append(cv2.contourArea(c))
            
    if (len(areas) == 0) or (np.max(areas) == 0):
        x = 0
        y = 0
        w = 0
        h = 0
        flag = True
    else:
        flag = False
        i = np.argmax(areas)
        [x,y,w,h] = cv2.boundingRect(cnts[i])
    #x+w, y+h
    return [x,y,w,h,flag]