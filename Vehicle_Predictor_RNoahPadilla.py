# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 23:28:28 2021
"""

'''
This program is a computer vision/ machine learning algorithm designed to predict a car.

The way the algorithm works - 

    1. Use a clustering algorithm (HOG's) to seperate the vehicle type (cars/vans vs trucks)
    2. Use the ORB + RANSAC algorithm to find the exact match
    
    
TODOS:
    
    [x] - Load and normalize the data from the train and test set (start small)
    [x] - Convert the train and test images into HOG's with an optimal number of bins
    [x] - Cluster the training set to seperate cars/vans and trucks //use 5% of training data 
    [x] - Test the cluster with the test data //use 20 images
    [] - Use ORB and RANSAC to find the make and model
'''



import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib.image as mpimg
import scipy.io
from scipy import signal
from sklearn.cluster import KMeans


from utils import *

#taken from exam 1
def histogram_of_gradients(img,bins=12):
    hog = []
    gray_conv = np.array([0.2989,0.5870,0.1140]).reshape(1,1,3)
    im = np.sum(img*gray_conv,axis=2)
    f1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    gh = signal.correlate2d(im, f1, mode='same')
    gv = signal.correlate2d(im, f1.T, mode='same')
    g_mag = np.sqrt(gv**2+gh**2)
    g_dir = np.arctan2(gv,gh)
    bin_size = 2*np.pi/bins
    g_dir_bin = ((g_dir+np.pi+bin_size/2)/bin_size).astype(np.int)%bins
    for i in range(bins):
        hog.append(np.sum(g_mag[g_dir_bin==i]))
    return np.array(hog)/im.shape[0]/im.shape[1]

if __name__ == "__main__":
    
    plt.close("all")
    
    image_dir_train = '.\\cars_train\\cars_train\\'
    image_dir_test = '.\\cars_test\\cars_test\\'
    image_files_train = os.listdir(image_dir_train)
    image_files_test= os.listdir(image_dir_test)
    
    #----------------------------- LOAD AND NORMALIZE IMAGES FROM TRAIN AND TEST SET --------------------------
    X_train = []
    X_test = []
    print("\n> Loading training data...")
    for i,image_file in enumerate(image_files_train[:250]):
        
        if i > 0 and i % 50 == 0:
            print("Pre-Processed", i , "images from train set")
        
        image = mpimg.imread(image_dir_train + image_file)
        
        #Normalize the image
        if np.amax(image)>1:
            image = (image/255).astype(np.float32)
            #print('Converted image to float')
        #print('Image shape',image.shape)#used for debugging
        #show_image(image,'Original image')#used for debugging
        X_train.append(np.array(image))
    
    print("\n> Loading test data...")
    for j,image_file in enumerate(image_files_test[:1000:50]):
        
        
        image = mpimg.imread(image_dir_test + image_file)
        
        #Normalize the image
        if np.amax(image)>1:
            image = (image/255).astype(np.float32)
            #print('Converted image to float')
        
        #print('Image shape',image.shape)
        #show_image(image,'Original image')
        
        X_test.append(np.array(image))
    
    
    #----------------------------- CONVERT TRAIN AND TEST IMAGES TO HOG --------------------------
    print("\n> Training set | Image representations -> HOG representations...")
    HOGS_train = []
    bins=128
    for i,x in enumerate(X_train):
        
        if i > 0 and i % 50 == 0:
            print("Converted", i , "images from train set")
        
        hog = histogram_of_gradients(x,bins=bins)
        HOGS_train.append(hog)#the training set are now represented using HOG
        #fig, ax = plt.subplots(2,figsize=(8,8))
        #fig.suptitle('Image ' + str(i))
        #ax[0].imshow(x)
        #ax[1].plot(np.arange(bins)*360/bins,hog)
        #ax[1].set_ylim([0,np.amax(hog)*1.1])
    
    print("\n> Test set | Image representations -> HOG representations...")
    HOGS_test = []
    for i,x in enumerate(X_test):
        
        if i > 0 and i % 50 == 0:
            print("Converted", i , "images from test set")
            
        hog = histogram_of_gradients(x,bins=bins)
        HOGS_test.append(hog)#the training set are now represented using HOG
        #fig, ax = plt.subplots(2,figsize=(8,8))
        #fig.suptitle('Image ' + str(i))
        #ax[0].imshow(x)
        #ax[1].plot(np.arange(bins)*360/bins,hog)
        #ax[1].set_ylim([0,np.amax(hog)*1.1])

    
    #-------------------------- PLACE EACH IMAGE (represented as HOG) INTO A CLUSTER -------------------
    clusters = 2 #cluster for cars/vans vs trucks
    print(" > Creating",clusters," clusters now...")
    #Create and fit data to 2 clusters (cars/vans, trucks)
    km = KMeans(n_clusters=clusters).fit(HOGS_train)
    print(" > Finished grouping data into ",clusters," clusters...")
    
    km_l = km.labels_
    #print('\nKMeans assignments/labels')
    #print(km_l)
    
    km_cc = km.cluster_centers_
    #print('\nFinal centroids:')
    #print(km_cc)
    
    km_ssd = km.inertia_ # sum of squared distance from each data point to its respective centroid
    #print('\nSSD: ')
    #print(km_ssd)
    
    
    '''
    #----------------------------------- PRINT VALUES FROM THEIR CLUSTERS -------------------
    pred = km.predict(HOGS_test)
    
    for i,x in enumerate(X_test):
        fig, ax = plt.subplots(2,figsize=(8,8))
        fig.suptitle('Image ' + str(i) + ' assigned to cluster: ' + str(pred[i]))
        ax[0].imshow(x)
        ax[1].plot(np.arange(bins)*360/bins,HOGS_test[i])
        ax[1].set_ylim([0,np.amax(HOGS_test[i])*1.1])
    '''
    
    #---------------------------- USE ORB AND RANSAC TO FIND IDENTICAL VEHICLE
    