from numpy import argmax, argmin, empty_like, dot
from sklearn.mixture import GMM
from scipy import misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow 

import warnings
warnings.filterwarnings("ignore")

def rgb2gray(rgb):
    return dot(rgb[...,:3], [0.299, 0.587, 0.114])


def run_GMM(rgb_pixel_values,n_components=15,covariance_type="full"):
    gmm = GMM(n_components,covariance_type="full").fit(rgb_pixel_values)
    gmm_labels = gmm.predict(rgb_pixel_values)
    
    return gmm, gmm_labels

def predict_pixel_GMM_cluster(pixel,gmm):
    
    pixel = pixel.reshape(1,3)
    return argmax(gmm.predict_proba(pixel))


def predict_pixel_GMM_color_MSE(pixel,gmm):
    pixel_MSE = ((gmm.means_ - pixel) ** 2).mean(axis=1)
    return gmm.means_[argmin(pixel_MSE)]


def predict_pixel_GMM_color_MSE_gray(pixel,gmm):
    pixel_MSE = ((rgb2gray(gmm.means_) - rgb2gray(pixel)) ** 2)
    return gmm.means_[argmin(pixel_MSE)]


def create_image_from_GMM(rgb_image_arr,gmm,downsample_factor = 0.1, show_original = False, gray = True):

    rgb_image_arr_downsampled = misc.imresize(rgb_image_arr,downsample_factor)

    
    if show_original == True:
        plt.figure(figsize=(10,10))
        imshow(rgb_image_arr_downsampled)

    gmm_image_prediction = empty_like(rgb_image_arr_downsampled)

    predict_pixel_GMM_color = predict_pixel_GMM_color_MSE
    if gray == True:
        predict_pixel_GMM_color = predict_pixel_GMM_color_MSE_gray
    
    for x in range(rgb_image_arr_downsampled.shape[0]):
        for y in range(rgb_image_arr_downsampled.shape[1]):
            gmm_image_prediction[x,y] = predict_pixel_GMM_color(rgb_image_arr_downsampled[x,y],gmm)

    plt.figure(figsize=(10,10))
    imshow(gmm_image_prediction)
    
    
