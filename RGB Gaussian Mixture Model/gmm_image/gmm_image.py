import numpy as np
# from sklearn.mixture import GMM

def predict_pixel_GMM_cluster(pixel,gmm):
    
    pixel = pixel.reshape(1,3)
    return np.argmax(gmm.predict_proba(pixel))

def predict_pixel_GMM_color(pixel,gmm):
    return gmm.means_[predict_pixel_GMM_cluster(pixel,gmm=gmm)]

def predict_pixel_GMM_color_MSE(pixel,gmm):
    pixel_MSE = ((gmm.means_ - pixel) ** 2).mean(axis=1)
    return gmm.means_[np.argmin(pixel_MSE)]

