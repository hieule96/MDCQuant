# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:37:58 2021

@author: LE Trung Hieu
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import Quadtreelib as qd
import Optimizer as Opt
import png
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

def get_PSNR(img1, img0,bittocoded=8):
    """
    Compute PSNR
    -----------
    Parameters:
    img1 : np_array
    img2 : np_array
    
    return
    ------------
    value : PSNR
    """ 
    max_i = np.max(img1)
    PSNR = 20 * np.log10(max_i) - 10 * np.log10(get_mse(img1, img0))
    return PSNR

def DisplayResult(LCU,img,title):
    import importlib
    import Quantlib as quant
    importlib.reload(__import__("Quantlib"))

    (imgdctDC,imgdctD1,imgdctD2,newimgDC,newimgD1,newimgD2)=quant.LCU_QuantDCTQP1QP2(LCU)
    newimgDC[newimgDC<-128] = -128
    newimgDC[newimgDC>127] = 127
    newimgDC = newimgDC + 128
    newimgDC = np.around(newimgDC)
    newimgDC = newimgDC.astype(np.uint8)
    newimgD1[newimgD1<-128] = -128
    newimgD1[newimgD1>127] = 127
    newimgD1 = newimgD1 + 128
    newimgD1 = np.around(newimgD1)
    newimgD1 = newimgD1.astype(np.uint8)
    newimgD2[newimgD2<-128] = -128
    newimgD2[newimgD2>127] = 127
    newimgD2 = newimgD2 + 128
    newimgD2 = np.around(newimgD2)
    newimgD2 = newimgD1.astype(np.uint8)    
    
    R1_AC = quant.LCU_CalRateQt(LCU,imgdctD1,AC_Remove = True)
    R2_AC = quant.LCU_CalRateQt(LCU,imgdctD2,AC_Remove = True)
    R1_DC = R2_DC = quant.LCU_CalRateQt_DC_DPCM(LCU,imgdctD1)
    R1 = R1_AC + R1_DC
    R2 = R2_AC + R2_DC
    D0 = np.around (mean_squared_error(img[:LCU.h,:LCU.w],newimgDC))
    D1 = np.around (mean_squared_error(img[:LCU.h,:LCU.w],newimgD1))
    D2 = np.around (mean_squared_error(img[:LCU.h,:LCU.w],newimgD2))
    PSNR0 = np.around (get_PSNR(img[:LCU.h,:LCU.w],newimgDC),4)
    PSNR1 = np.around (get_PSNR(img[:LCU.h,:LCU.w],newimgD1),4)
    PSNR2 = np.around (get_PSNR(img[:LCU.h,:LCU.w],newimgD2),4)
    
    # (newimg8x8, imgdct8x8, Rate8x8) = quant.QuantDCT8x8Rate(img,img.shape[0], img.shape[1],R1+R2,1000)
    # Rate8x8 = Rate8x8/(imgdct8x8.shape[0]*imgdct8x8.shape[1])
    print ("R0 Theorique (bytes):" + str(quant.LCU_CalRateQt(LCU,imgdctDC)/8)+"bytes")
    print ("R1 Theorique (bytes):" + str(R1/8)+"bytes")
    print ("R2 Theorique (bytes):" + str(R1/8)+"bytes")
    
    R1_AC = np.around(R1_AC/(img.shape[0]*img.shape[1]),3)
    R2_AC = np.around(R2_AC/(img.shape[0]*img.shape[1]),3)
    R1 = np.around(R1/(img.shape[0]*img.shape[1]),3)
    R2 = np.around(R2/(img.shape[0]*img.shape[1]),3)
    R0 = np.around(R1+R2,3)
    R0_AC = np.around(R1_AC+R2_AC,3)
    plt.figure(4,figsize=(20,20))
    plt.subplot(321), plt.imshow(newimgD1,cmap='gray'), plt.title("D1 MSE: %s PNSR: %s R1_AC: %s R1_T %s" %(D1,PSNR1,R1_AC,R1))
    plt.subplot(322), plt.imshow(newimgD2,cmap='gray'), plt.title("D2 MSE: %s PNSR: %s R2_AC: %s R2_T %s" %(D2,PSNR2,R2_AC,R2))
    plt.subplot(323), plt.imshow(newimgDC,cmap='gray'), plt.title("D0 MSE: %s PNSR: %s R0_AC: %s R0_T %s" %(D0,PSNR0,R0_AC,R0))
    plt.show()
    return PSNR0,PSNR1,PSNR2,R0,R1,R2
    # plt.subplot(324), plt.imshow(newimg8x8,cmap='gray'), plt.title("JPEG MSE:" +str(np.around (mean_squared_error(img[:,:],newimg8x8))) + " PNSR: "+str(np.around (Optimizer_Laplace.get_PSNR(img[:,:],newimg8x8))) + " R8x8:" + str(np.around(Rate8x8,3))+"bits/pixel")

def DisplayQuantizationMap(i_QuantizationMap1,i_QuantizationMap2,title):
    fig= plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(cv2.cvtColor(i_QuantizationMap1, cv2.COLOR_BGR2RGB))
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(cv2.cvtColor(i_QuantizationMap2, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def DisplayEnergyMap(i_EnergyMap,title):
    fig= plt.figure(figsize=(20, 20))
    rgb = cv2.cvtColor(i_EnergyMap, cv2.COLOR_BGR2RGB)
    plt.title(title)
    plt.imshow(rgb)
    plt.show()
    
def bgr(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return b,g,r

img_path = "news_cif/news_cif1.bmp"
CTU_path = "news_cif/decoder_cupu.txt"
(LCU,imgY) = qd.ImportQuadTreeYchannel(img_path,CTU_path,29)
LCU.convert_Tree_childrenArray()
LCU.remove_bord_elements(LCU.w,LCU.h)
LCU.Init_aki()
LCU.merge_CTU()
# qd.printI(LCU.render_img(thickness=1, color=(255,255,255)),'Import CTUs from files')
Opt.Optimizer_curvefitting.initCoefficient(LCU)
# LCU.ExportParamtertoCSV(img_path)
GlobalParam = Opt.OptimizerParameterGeneric(lam1_min = 0,lam2_min=0,mu1=0.1,mu2=1,n0=0.5,QPmax=100,LCU=LCU,Rt=0.01,Dm=200,deltaRateTarget=.01)
Oj = Opt.Optimizer_curvefitting(GlobalParam)

Rt_possible =  [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
PSNR0HEVC_array,PSNR1HEVC_array,PSNR2HEVC_array,R0HEVC_array,R1HEVC_array,R2HEVC_array = [],[],[],[],[],[]
for Rt in Rt_possible:
    Oj.setRt(Rt)
    print (Oj.globalparam.Rt)
    (Q1,Q2,D1_est,D2_est,R1_est,R2_est) = Oj.optimize_LCU() 
    PSNR0,PSNR1,PSNR2,R0,R1,R2 = DisplayResult(LCU,imgY,"Result" )
    PSNR0HEVC_array.append(PSNR0)
    PSNR1HEVC_array.append(PSNR1)
    PSNR2HEVC_array.append(PSNR2)
    R0HEVC_array.append(R0)
    R1HEVC_array.append(R1)
    R2HEVC_array.append(R2)
# LCU=0
# img=0
# img_path = "lena512color.tiff"
# threshold = 1800
# #width and height must be identical and divisible by 64
# (LCU,img) = qd.ComputeQuadTreeYChannel(img_path,threshold=threshold,minPixelSize=8,x0=0,y0=0,CTU_size_h = 512,CTU_size_w = 512)
# Opt.Optimizer_curvefitting.initCoefficient(LCU)
# LCU.ExportParamtertoCSV(img_path)
# GlobalParam = Opt.OptimizerParameterGeneric(lam1_min = 0,lam2_min=0,mu1=0.1,mu2=1,n0=0.5,QPmax=100,LCU=LCU,Rt=0.01,Dm=200,deltaRateTarget=.001)
# Oj = Opt.Optimizer_curvefitting(GlobalParam)
# Rt_possible = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# PSNR0_array,PSNR1_array,PSNR2_array,R0_array,R1_array,R2_array = [],[],[],[],[],[]
# for Rt in Rt_possible:
#     Oj.setRt(Rt)
#     print (Oj.globalparam.Rt)
#     (Q1,Q2,D1_est,D2_est,R1_est,R2_est) = Oj.optimize_LCU() 
#     PSNR0,PSNR1,PSNR2,R0,R1,R2 = DisplayResult(LCU, "Result" )
#     PSNR0_array.append(PSNR0)
#     PSNR1_array.append(PSNR1)
#     PSNR2_array.append(PSNR2)
#     R0_array.append(R0)
#     R1_array.append(R1)
#     R2_array.append(R2)
    
plt.plot(R0HEVC_array,PSNR0HEVC_array,'r-',label="PSNR0")
plt.plot(R0HEVC_array,PSNR1HEVC_array,'b-',label="PSNR1")
plt.plot(R0HEVC_array,PSNR2HEVC_array,'g-',label="PSNR2")
plt.xlabel("Rate[bits/pixel]")
plt.ylabel("PSNR[dB]")
plt.legend()
plt.show()
# plt.plot(R0_array,PSNR0_array,'r-',label="PSNR0")
# plt.plot(R0_array,PSNR1_array,'b-',label="PSNR1")
# plt.plot(R0_array,PSNR2_array,'g-',label="PSNR2")
# plt.xlabel("Rate[bits/pixel]")
# plt.ylabel("PSNR[dB]")
# plt.legend()
# plt.show()

# with open('Result_HEVC_EnergyMap/Performance%s_HEVC_vs_Threshold%s.csv' %(img_path,threshold),'w') as file:
#     file.write("R0_HEVC,R0,PSNR0HEVC,PSNR0,PSNR1HEVC,PSNR1,PSNR2HEVC,PSNR2" + '\n')
#     for i in range (len(R0_array)):
#             file.write("%s,%s,%s,%s,%s,%s,%s,%s,%s" %(R0HEVC_array[i],R0_array[i],PSNR0HEVC_array[i],PSNR0_array[i],PSNR1HEVC_array[i],PSNR1_array[i],PSNR2HEVC_array[i],PSNR2_array[i],Rt_possible[i]))
#             file.write("\n")
# file.close()

# R0_array = np.multiply(R0_array,img.shape[0]*img.shape[1])
# plt.plot(R0_array,PSNR0_array,'r-',label="PSNR0")
# plt.plot(R0_array,PSNR1_array,'b-',label="PSNR1")
# plt.plot(R0_array,PSNR2_array,'g-',label="PSNR2")
# plt.xlabel("Rate[bits/pixel]")
# plt.ylabel("PSNR[dB]")
# plt.legend()
# plt.show()



