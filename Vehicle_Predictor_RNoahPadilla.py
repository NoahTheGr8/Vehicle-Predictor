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
    
    for image_file in image_files_train[:10]:

        image = mpimg.imread(image_dir_train + image_file)
        print('Image shape',image.shape)
        show_image(image,'Original image')