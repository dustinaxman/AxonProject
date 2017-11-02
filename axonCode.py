#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:46:57 2017

@author: deaxman
"""
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


#%% Set up data, extract features
from skimage.feature import hog
from skimage import data, color, exposure
fileDict=scipy.io.loadmat()
fileDict2=scipy.io.loadmat()
fileName=''
fileName2=''
data=fileDict[fileName]
y=fileDict2[fileName2]


fd, hog_image = hog(data[0,:,:], orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

X=np.zeros((X.shape[0],len(fd)))
for i in range(X.shape[0]):
    fd, hog_image = hog(data[i,:,:], orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True)
    X[i,:]=fd
    
#%%
percentTest=0.2
trainIdx=np.random.choice(X.shape[0],size=int(np.round(X.shape[0]*(1-percentTest))),replace=False)
testIdx=np.setdiff1d(np.arange(X.shape[0]),trainIdx)
Xtr=X[trainIdx,:]
Xte=X[testIdx,:]
ytr=y[trainIdx]
yte=y[testIdx]


#%%Random forest regression
#Typically the number of features to use on each split is d/3 for regression in RF models.  We will xval this anyway just to be sure.
#Choosing the number of trees is a little tricky.  Many papers say that RF models don't overfit as number of trees increases.
#Mark R. Segal (April 14 2004. "Machine Learning Benchmarks and Random Forest Regression." Center for Bioinformatics & Molecular Biostatistics)
# says that they can overfit for extremely noisy datasets.  I'm going to xvalidate this parameter just in case, though I expect it to be on the high end of whatever range we give it.
from sklearn.ensemble import RandomForestRegressor
#Train/x-Val

#This is what I wanted to do if I had more computation power
#param_grid = {'n_estimators': np.array([10,20,50,100]),
#              'max_features': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])}

param_grid = {'n_estimators': np.array([20,50]),
              'max_features': np.array([0.2, 0.3, 0.4])}


gridSearchObject = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
fig = plt.figure()
ax = plt.axes()
N, train_lc, val_lc = learning_curve(gridSearchObject,
                                         Xtr, ytr, cv=5,
                                         train_sizes=np.linspace(0.3, 1, 25))
ax.plot(N,train_lc)
ax.plot(N,val_lc)

gridSearchObject.fit(Xtr,ytr)


print(gridSearchObject.best_params_)

bestModel=gridSearchObject.best_estimator_


#Predict (interpret) mean squared error and r2
ypred = bestModel.fit(Xtr, ytr).predict(Xte)
mse =  mean_squared_error(yte, ypred)
r2 =  r2_score(yte, ypred)
