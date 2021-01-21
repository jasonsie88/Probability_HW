#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample code of HW4, Problem 3
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
import math
from scipy import linalg

myfile = open('hw4_p3_data.pickle', 'rb')
mydict = pickle.load(myfile)

X_train = mydict['X_train']
X_test = mydict['X_test']
Y_train = mydict['Y_train']
Y_test = mydict['Y_test']

predictive_mean = np.empty(X_test.shape[0])
predictive_std = np.empty(X_test.shape[0])

sigma = 0.1
sigma_f = 1.0
#ls = 0.06
ls=0.2
#-------- Your code (~10 lines) ---------
cov=[[0]*25 for i in range(25)]
tmp=[[0]*25 for i in range(25)]
I=[[0]*25 for i in range(25)]
k=[[0]*25 for i in range(25)]

for i in range(0,25):
    for j in range(0,25):
        if(i==j):
            I[i][j]=1
        else:
            I[i][j]=0
            
for i in range(0,25):
    for j in range(0,25):
        cov[i][j]=(sigma_f**2)*math.exp(-((X_train[i]-X_train[j])**2)/2*(ls**2))+I[i][j]*sigma**2
    
for i in range(0,25):
    for j in range(0,25):
        I[i][j]=I[i][j]*(sigma**2)
        
for i in range(0,25):
    for j in range(0,25):
        tmp[i][j]=I[i][j]+cov[i][j]
k=np.linalg.inv(tmp)

for i in range(0,75):
    xk=np.empty(X_train.shape[0])
    for j in range(0,25):
        if(i<25):
            xk[j]=(sigma_f**2)*math.exp(-((X_train[j]-X_test[i])**2)/2*(ls**2))+I[i][j]
        elif(i<50):
            xk[j]=(sigma_f**2)*math.exp(-((X_train[j]-X_test[i])**2)/2*(ls**2))+I[i-25][j]
        elif(i<75):
             xk[j]=(sigma_f**2)*math.exp(-((X_train[j]-X_test[i])**2)/2*(ls**2))+I[i-50][j]
    predictive_mean[i]=xk.dot(k).dot(Y_train.T)
for i in range(0,75):
    xk=np.empty(X_train.shape[0])
    for j in range(0,25):
        if(i<25):
            xk[j]=(sigma_f**2)*math.exp(-((X_train[j]-X_test[i])**2)/2*(ls**2))+I[i][j]
        elif(i<50):
            xk[j]=(sigma_f**2)*math.exp(-((X_train[j]-X_test[i])**2)/2*(ls**2))+I[i-25][j]
        elif(i<75):
             xk[j]=(sigma_f**2)*math.exp(-((X_train[j]-X_test[i])**2)/2*(ls**2))+I[i-50][j]
    predictive_std[i]=(xk.dot(k).dot(xk.T))

#---------- End of your code -----------

# Optional: Visualize the training data, testing data, and predictive distributions
fig = plt.figure()
plt.plot(X_train, Y_train, linestyle='', color='b', markersize=3, marker='+',label="Training data")
plt.plot(X_test, Y_test, linestyle='', color='orange', markersize=2, marker='^',label="Testing data")
plt.plot(X_test, predictive_mean, linestyle=':', color='green')
plt.fill_between(X_test.flatten(), predictive_mean - predictive_std, predictive_mean + predictive_std, color='green', alpha=0.13)
plt.fill_between(X_test.flatten(), predictive_mean - 2*predictive_std, predictive_mean + 2*predictive_std, color='green', alpha=0.07)
plt.fill_between(X_test.flatten(), predictive_mean - 3*predictive_std, predictive_mean + 3*predictive_std, color='green', alpha=0.04)
plt.xlabel("X")
plt.ylabel("Y")
