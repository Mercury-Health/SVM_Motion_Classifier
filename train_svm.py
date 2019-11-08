# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 09:12:09 2019

@author: Michael K
"""

import scipy
import libsvm.python.svmutil as svmutil
import numpy as np
import os
from sklearn.metrics import confusion_matrix


def make_vectors(list_obj,k):
    out = []
    for el in list_obj:
        x_p = el[0]
        y_p = el[1]
        if(len(x_p) >= k):
            outvecs = np.concatenate([x_p[0:k],y_p[0:k]])
            out = np.concatenate([out,outvecs])
    x_dim = 2*k
    return np.reshape(out,(x_dim,int(out.shape[0]/x_dim)))

def handle_class(dat,k,holdout):
    x = make_vectors(dat,k)
    y = np.ones(x.shape[1])
    inds = np.random.permutation(x.shape[1])
    use = inds[0:int(np.floor(len(inds)*(1-holdout)))]
    test = inds[int(np.floor(len(inds)*(1-holdout))):len(inds)]
    return y[use],x[:,use],y[test],x[:,test]
    
def form_svm_input(dat_list,k,holdout):
    y = np.asarray([])
    x = np.empty([k*2,0])
    y_test = np.asarray([])
    x_test = np.empty([k*2,0])
    count = 1
    for dat in dat_list:
        y_temp, x_temp, y_test_temp, x_test_temp = handle_class(dat,k,holdout)
        y_temp = count*y_temp
        y_test_temp = count*y_test_temp
        count = count + 1
        y = np.concatenate([y,y_temp])
        x = np.concatenate([x,x_temp],1)
        y_test = np.concatenate([y_test,y_test_temp])
        x_test = np.concatenate([x_test,x_test_temp],1)
    return scipy.asarray(y), scipy.asarray(x), scipy.asarray(y_test), scipy.asarray(x_test)
    
        
base_dir = "C:/Users/Michael K/Desktop/real/"

files = os.listdir(base_dir)
files = [el for el in files if '.npy' in el]
files = [el for el in files if 'Val' not in el]
dat_list = [np.load(base_dir+el,allow_pickle = True) for el in files]

k = 20
holdout = 0
y,x,y_test,x_test = form_svm_input(dat_list,k,holdout)

# train SVM
#y, x = scipy.asarray(Y), scipy.asarray(X)
#y_test, x_test = scipy.asarray(Y_test), scipy.asarray(X_test)
x = scipy.transpose(x)
#x_test = scipy.transpose(x_test)
prob = svmutil.svm_problem(y,x)
#param = svmutil.svm_parameter('-s 1 -t 1 -c 1 -n .05 -b 0 -m 10240') #1 works best, cost of 100 seems good (.000001 multiclass)
param = svmutil.svm_parameter('-t 1 -c .000001 -b 0')
m = svmutil.svm_train(prob,param)

#label = svmutil.svm_predict(y_test, x_test, m)

files = os.listdir(base_dir)
files = [el for el in files if '.npy' in el]
files = [el for el in files if 'Val' in el]
dat_list = [np.load(base_dir+el,allow_pickle = True) for el in files]

y_test,x_test,dc1, dc2 = form_svm_input(dat_list,k,0)
x_test = scipy.transpose(x_test)
label = svmutil.svm_predict(y_test, x_test, m)

print(confusion_matrix(label[0],y_test))

