# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:46:52 2021

@author: hieu1
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 19:24:01 2021

@author: hieu1
"""
import numpy as np
import scipy.optimize
import Transform as tf
from matplotlib import pyplot as plt
import Quadtreelib as qd
import Quantlib as quant
import logging
from scipy.optimize import curve_fit
import skimage.metrics
import pdb
import Optimizer as opt
def get_mse(img1, img2):
    """
    Compute MSE
    -----------
    Parameters:
    img1 : np_array
    img2 : np_array
    
    return
    ------------
    value : MSE
    """ 
    diff = np.subtract(img1[:], img2[:])
    MSE = np.square(diff).mean()
    return MSE

def neg_exponential_function(x,a,b,c):
    return a*np.exp(-b*x)+c
def exponential_function(x,a,b):
    return a*np.exp(b*x)
def cubic_function(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d 
def linear_function(x,a,b):
    return a*x + b
def derivate_tanh_exp(x,a1,b1,a2,b2):
    return -((a2*b2)/(a1*b1))*(np.exp(b1*x)/(np.cosh(b2*x)**2))
def derivate_exp_lin(x,a1,b1,a2):
    return a1*b1*np.exp(x)/a2
def rateDistortionRealACNode(LCU,node,QP):
    img_cu = node.get_points(LCU.img)
    img_dc = img_cu.mean()
    img_dct_cu_AC = tf.dct(img_cu-img_dc)
    # img_dct_cu_AC = img_dct_cu_AC[:,:]
    # img_dct = tf.dct(img_cu)
    # pdb.set_trace()
    # mQ_AC = np.delete(mQ,(0,0))
    imgdctQ = quant.quantCUSimple(img_dct_cu_AC, QP)
    imgdctdQ = quant.deQuantCUSimple(imgdctQ, QP)

    #Calculate entropy
    unique,counts =np.unique(imgdctQ,return_counts=True)
    probalitity = counts/counts.sum()
    entropy = -np.sum(probalitity*np.log2(probalitity))
    #Calculate MSE
    img_rec_2D = tf.idct(imgdctdQ) + img_dc
    # pdb.set_trace()
    img_rec_2D [img_rec_2D <-128] = -128
    img_rec_2D [img_rec_2D >127] = 127
    mse = skimage.metrics.mean_squared_error(img_rec_2D,img_cu)
    # pdb.set_trace()
    return mse,entropy
def nodeACStd(LCU,node):
    img_cu = node.get_points(LCU.img)
    img_dct_cu_AC = tf.dct(img_cu)
    img_dct_cu_AC = np.delete(img_dct_cu_AC,(0,0))
    return img_dct_cu_AC.std()
def get_rsquare(data_fit,real_data):
    residuals = real_data - data_fit
    ss_tot = np.sum((real_data - np.mean(real_data))**2)
    ss_res = np.sum(residuals**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# def plot_Spectrum_DCT(LCU,node,Q):
#     img_cu = node.get_points(LCU.img)
#     img_dct_cu = tf.dct(img_cu)
#     freq = []
#     for i in range (len(img_dct_cu)):
#         freq = img_dct_cu[i][j]
        
(LCU,img) = qd.ComputeQuadTreeYChannel("lena512color.tiff",threshold=180,minPixelSize=8,x0=0,y0=0,CTU_size_h = 512,CTU_size_w = 512)
i = 0
Rsquare = []
# for CTU in LCU.CTUs:
Qvalue = np.arange(0,51,1)
#     j = 0
#     for cu in CTU:
#         sigma = nodeACStd(LCU,cu)
mse_Q = []
entropy_Q = []
for Q in Qvalue:
    mse,entropy = rateDistortionRealACNode(LCU,LCU.CTUs[0][0],Q)
    mse_Q.append(mse)
    entropy_Q.append(entropy)
    # if (mse >= 1):
    #     break

Qvalue = Qvalue[:Q+1]
# poptR, pcov = curve_fit(neg_exponential_function, Qvalue, entropy_Q)
# entropy_fit = neg_exponential_function(Qvalue, *poptR)

# poptD, pcov = curve_fit(exponential_function, Qvalue, mse_Q)
# mse_fit = exponential_function(Qvalue, *poptD)


entropy_Q = np.array(entropy_Q)
mse_Q = np.array(mse_Q)
# poptDR, pcovDR = curve_fit(neg_exponential_function, entropy_Q, mse_Q)
# DR_fit = neg_exponential_function(entropy_Q, *poptDR)
# plt.plot(entropy_Q, DR_fit,color="r")
entropyCvx = opt.ConvexHull(entropy_Q[:-10], Qvalue[:-10])
entropy_Quni,QP_unientropy = entropyCvx.getunique()
logentropy_Quni = np.log(entropy_Quni)

a,b,r,_,_ = scipy.stats.linregress(QP_unientropy,logentropy_Quni)
Rcoeff, pcovR = curve_fit(exponential_function, QP_unientropy,entropy_Quni,p0=[np.exp(b),a])
entropy_fit = exponential_function(QP_unientropy,*Rcoeff)
plt.scatter(QP_unientropy,entropy_Quni)
plt.plot(QP_unientropy,entropy_fit)
plt.show()


MSECvx = opt.ConvexHull(mse_Q, Qvalue)
MSE_uni,QP_unimse = MSECvx.getunique()
logMSEuni = np.log(MSE_uni)

plt.scatter(entropy_Q,mse_Q)
plt.show()
plt.scatter(QP_unimse,MSE_uni)
plt.plot(Qvalue,mse_Q)
plt.show()
# plt.scatter(QP_unientropy,entropy_Quni)
# plt.plot(Qvalue,entropy_Q)
# plt.show()
        # Rsquare.append([i,j,get_rsquare(mse_fit,mse_Q),get_rsquare(entropy_fit,entropy_Q),get_rsquare(DR_fit,mse_Q)])
    #     print (i,j)
    #     j = j+1
    # i = i+1
