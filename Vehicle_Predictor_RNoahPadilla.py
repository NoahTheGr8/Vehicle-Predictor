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
import time 

from skimage import transform as tf
from matplotlib import transforms
import cv2
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

def display_correspondences(im0,im1,pts0,pts1,clr_str = 'rgbycmwk'):
    canvas_shape = (max(im0.shape[0],im1.shape[0]),im0.shape[1]+im1.shape[1],3)
    canvas = np.zeros(canvas_shape,dtype=type(im0[0,0,0]))
    canvas[:im0.shape[0],:im0.shape[1]] = im0
    canvas[:im1.shape[0],im0.shape[1]:canvas.shape[1]]= im1
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(canvas)
    ax.axis('off')
    pts2 = pts1+np.array([im0.shape[1],0])
    for i in range(pts0.shape[0]):
        ax.plot([pts0[i,0],pts2[i,0]],[pts0[i,1],pts2[i,1]],color=clr_str[i%len(clr_str)],linewidth=1.0)
    fig.suptitle('Point correpondences', fontsize=16)

def select_matches_ransac(pts0, pts1):
    H, mask = cv2.findHomography(pts0.reshape(-1,1,2), pts1.reshape(-1,1,2), cv2.RANSAC,5.0)
    choice = np.where(mask.reshape(-1) ==1)[0]
    return pts0[choice], pts1[choice]

if __name__ == "__main__":
    
    plt.close("all")
    
    image_dir_train = '.\\cars_train\\cars_train\\'
    image_dir_test = '.\\cars_test\\cars_test\\'
    image_files_train = os.listdir(image_dir_train)
    image_files_test= os.listdir(image_dir_test)
    
    #----------------------------- LOAD AND NORMALIZE IMAGES FROM TRAIN AND TEST SET --------------------------
    
    #TODO - MAKE SURE LATER ON YOU REMOVE THE LABELED DATA THAT CORRESPONDED TO THE IMAGES THAT DIDNT HAVE 3 LAYERS
    X_train = []
    X_test = []
    train_limit = 5398
    print("\n> Loading training data...")
    for i,image_file in enumerate(image_files_train[:train_limit]):
        
        if i > 0 and i % 1000 == 0:
            print("Pre-Processed", i , "images from train set")
        
        image = mpimg.imread(image_dir_train + image_file)
        
        if len(image.shape) == 3:
            X_train.append(np.array(image))
    
    print("\n> Loading test data...")
    for j,image_file in enumerate(image_files_test[:1000:10]):
        
        image = mpimg.imread(image_dir_test + image_file)
        
        #checks that the image has 3 layers - discard images with only 2
        if len(image.shape) == 3:
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
    
    
    
    #----------------------------------- PREDICT WHAT CLUSTERS THE TEST SAMPLES ARE IN -------------------
    pred = km.predict(HOGS_test)
    
    orb = cv2.ORB_create()
    '''
    image_dir_train = '.\\cars_train\\cars_train\\'
    image_dir_test = '.\\cars_test\\cars_test\\'
    image_files_train = os.listdir(image_dir_train)
    image_files_test= os.listdir(image_dir_test)
    '''
    
    for i,x_test_sample in enumerate(X_test[:3]):
        fig, ax = plt.subplots(2,figsize=(8,8))
        fig.suptitle('Test image ' + str(i) + ' assigned to cluster: ' + str(pred[i]))
        
        keypoints1, descriptors1 = orb.detectAndCompute(np.asarray(x_test_sample, np.uint8),mask=None)
        
        best_p0 = None #stores the best point matches on image 1
        best_p1 = None #stores the best point matches on image 2
        max_feature_matches = 0 #will be used to count the matched features from two images
        matched_vehicle = None # stores the array containing the matched vehicle
        
        for j,x_train_sample_label in enumerate(km_l):
            if j%200==0:
                print("Looked through",j, "images - still looking for best match")
            #if we look at the cluster which x_test_sample defines itself then we can use ORB +RANSAC to look within cluster to find similar vehicle
            if pred[i] == x_train_sample_label:
                #print("x_test_sample: ", x_test_sample.shape, "| X_train[i] shape:", X_train[j].shape)
                #print("pred:" , pred[i], "| x_train_sample_label: ", x_train_sample_label)
                
                #---------------------------- USE ORB AND RANSAC TO FIND IDENTICAL VEHICLE
                keypoints2, descriptors2 = orb.detectAndCompute(np.asarray(X_train[j], np.uint8),mask=None)
                
                # Create BFMatcher object
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(descriptors1,descriptors2)
                
                #find the subest of these matches that can be explained using the ransac | for every image find the best matching image
            
                # Extract data from orb objects and matcher
                dist = np.array([m.distance for m in matches])
                ind1 = np.array([m.queryIdx for m in matches])
                ind2 = np.array([m.trainIdx for m in matches])
                
                #TWEAKED BY NOAH - NEEDED TO CHANGE IF GOING THROUGH A LOOP
                k1 = np.array([p.pt for p in keypoints1])
                k2 = np.array([p.pt for p in keypoints2])
                k1 = k1[ind1]
                k2 = k2[ind2]
                # keypoints1[i] and keypoints2[i] are a match
                
                #---------------------BEST MATCH POINTS FROM BOTH IMAGES
                p0, p1 = select_matches_ransac( k1, k2 )
                if (p0.shape[0] > max_feature_matches):
                    best_p0 = p0
                    best_p1 = p1
                    max_feature_matches = p0.shape[0]
                    matched_vehicle = X_train[j]
        
        display_correspondences(x_test_sample,matched_vehicle, best_p0, best_p1)
        ax[0].imshow(x_test_sample)
        ax[1].imshow(matched_vehicle)
        #ax[1].plot(np.arange(bins)*360/bins,HOGS_test[i])
        #ax[1].set_ylim([0,np.amax(HOGS_test[i])*1.1])
        #break