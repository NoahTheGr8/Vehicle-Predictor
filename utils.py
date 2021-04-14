import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import pandas as pd
from skimage import exposure
from skimage import img_as_float

def array_to_excel(A,filename):
    df = pd.DataFrame()
    for c in range(A.shape[1]):
        df[c] = A[:,c]
    df.to_excel(filename+".xlsx",header=False, index=False)

def integral_image(image):
    if len(image.shape)>2:
        image = gray_level(image)
    if 'int' in str(type(image[0,0])):
        integral_image = np.zeros((image.shape[0]+1,image.shape[1]+1),dtype=np.int32)
    else:
        integral_image = np.zeros((image.shape[0]+1,image.shape[1]+1),dtype=np.float32)
    integral_image[1:,1:] = np.cumsum(np.cumsum(image,axis=0),axis=1)
    return integral_image

def region_sums(image,reg_rows,reg_cols):
    S = integral_image(image)
    return S[reg_rows:,reg_cols:] - S[reg_rows:,:-reg_cols] - S[:-reg_rows,reg_cols:] + S[:-reg_rows,:-reg_cols]

def show_image(image,title='',save_im=False,filename=None):
    # Display image in new window
    fig, ax = plt.subplots()
    ax.imshow(image,cmap='gray')
    ax.axis('off')
    ax.set_title(title)
    if save_im:
        if filename==None:
            filename=title
        fig.savefig(filename+'.jpg',bbox_inches='tight', pad_inches=0.1)
    return fig, ax

def show_images(images,titles=None,fig_title='', save_im=False,filename=None):
    if titles==None:
        titles = ['' for i in range(len(images))]
    # Display image in new window
    fig, ax = plt.subplots(1,len(images),figsize=(12, 4))
    for i in range(len(images)):
        ax[i].imshow(images[i],cmap='gray')
        ax[i].axis('off')
        ax[i].set_title(titles[i])
    fig.suptitle(fig_title, fontsize=16)
    if save_im:
        if filename==None:
            filename='show_images'
        fig.savefig(filename+'.jpg',bbox_inches='tight', pad_inches=0.1)
    return fig, ax

def color_index(image,index):
    return 3*image[:,:,index] - np.sum(image,axis=2)

def subsample(image,r,c):
    return image[::r,::c]

def gray_level(image):
    gray_conv = np.array([0.2989,0.5870,0.1140]).reshape(1,1,3)
    return np.sum(image*gray_conv,axis=2)

def negative_gray_level(image):
    return 1 - gray_level(image)

def vert_edges(gray_image):
    if len(gray_image.shape)>2:
        gray_image = gray_level(gray_image)
    edges =  np.zeros_like(gray_image)
    edges[:,:-1] = gray_image[:,:-1] -  gray_image[:,1:]
    return edges

def hor_edges(gray_image):
    if len(gray_image.shape)>2:
        gray_image = gray_level(gray_image)
    edges =  np.zeros_like(gray_image)
    edges[:-1] = gray_image[:-1] -  gray_image[1:]
    return edges

def mirror(image):
    return image[:,::-1]

def upside_down(image):
    return image[::-1]

def make_box(x, y, dx, dy):
    # Returns coordinates of box given upper left corner (x,y) and sizes (dx, dy)
    # Notice x corresponds to image columns and y to image rows
    xs = x + np.array([0,1,1,0,0])*dx - 0.5
    ys = y + np.array([0,0,1,1,0])*dy - 0.5
    return xs,ys

def brightest_region(image,reg_rows,reg_cols):
    rs = region_sums(image,reg_rows,reg_cols)
    brightest = np.argmax(rs)
    brightest_col = brightest%rs.shape[1]
    brightest_row = brightest//rs.shape[1]
    return brightest_row, brightest_col, rs[brightest_row, brightest_col]

#------------------------------TODO
#Display the image of a adaptive histogram filter applied
def plot_img(image, axes, bins=256):
    image = img_as_float(image)
    ax_img = axes
 
    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    return ax_img

