# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 23:28:28 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from utils import *

if __name__ == "__main__":
    
    plt.close("all")
    
    image_dir_train = '.\\cars_train\\cars_train\\'
    image_dir_test = '.\\cars_test\\cars_test\\'
    image_files_train = os.listdir(image_dir_train)
    image_files_test= os.listdir(image_dir_test)
    
    #Load the train and test images into X | I will split how I want not 5050
    X = []
    
    for i,image_file in enumerate(image_files_train[:10]):
        
        if i > 0 and i % 1000 == 0:
            print("rocessed", i , "images")
        
        image = mpimg.imread(image_dir_train + image_file)
        
        #Normalize the image
        if np.amax(image)>1:
            image = (image/255).astype(np.float32)
            #print('Converted image to float')
        
        #print('Image shape',image.shape)
        #show_image(image,'Original image')
        
        X.append(np.array(image))
    
    for j,image_file in enumerate(image_files_test[:10]):
        
        if j > 0 and j % 1000 == 0:
            print("Processed", j , "images")
        
        image = mpimg.imread(image_dir_test + image_file)
        
        #Normalize the image
        if np.amax(image)>1:
            image = (image/255).astype(np.float32)
            #print('Converted image to float')
        
        #print('Image shape',image.shape)
        #show_image(image,'Original image')
        
        X.append(np.array(image))
        
    #X = np.array(X , dtype=object)
    
    '''
    TODO - Get the labels of the vehicles and put them in 'y' in their respective positions so that we could split and train
    '''
    
    
    
    
    
    
    
    
    
    
    
    