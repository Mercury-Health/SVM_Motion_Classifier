# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:03:28 2019

@author: Michael K
"""

import os
from Libraries.make_frame import make_frame

if not os.path.exists('data'):
    os.mkdir('data')
    os.mkdir('data/up')
    os.mkdir('data/down')
    
base_dir = 'C:/Users/Michael K/Desktop/real'
#dirs = os.listdir(base_dir)
dirs = ['Falling_all']
for fol in dirs:
    full_path = base_dir+'/'+fol
    count = 0
    for el in os.listdir(full_path):
        if 'avi' in el:
            make_frame(full_path+'/'+el,'data/up/'+str(count),0)
            make_frame(full_path+'/'+el,'data/down/'+str(count),29)
            count = count + 1
            print(count)