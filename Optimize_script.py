# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 00:24:42 2021

@author: hieu1
"""

import cv2
from matplotlib import pyplot as plt
import random
import math
import numpy as np
import Quadtreelib as Qd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import Quadtreelib as qd
import Optimizer_model_laplace as Optimizer_Laplace
import Transform as tf





def DisplayResult(LCU,img,title):
    import importlib
    import Quantlib as quant
    importlib.reload(__import__("Quantlib"))
    
    (imgdctDC,imgdctD1,imgdctD2,newimgDC,newimgD1,newimgD2)=quant.LCU_QuantDCTQ1Q2(LCU)
    R1_AC = quant.LCU_CalRateQt(LCU,imgdctD1,AC_Remove = True)
    R2_AC = quant.LCU_CalRateQt(LCU,imgdctD2,AC_Remove = True)
    R1_DC = R2_DC = quant.LCU_CalRateQt_DC_DPCM(LCU,imgdctD1)
    R1 = R1_AC+R1_DC
    R2 = R2_AC+R2_DC
    D0 = np.around (mean_squared_error(img[:,:],newimgDC))
    D1 = np.around (mean_squared_error(img[:,:],newimgD1))
    D2 = np.around (mean_squared_error(img[:,:],newimgD2))
    PSNR0 = np.around (Optimizer_Laplace.get_PSNR(img[:,:],newimgDC))
    PSNR1 = np.around (Optimizer_Laplace.get_PSNR(img[:,:],newimgD1))
    PSNR2 = np.around (Optimizer_Laplace.get_PSNR(img[:,:],newimgD2))
    
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

def plotEnergymap(img_path,LCU,min_value,max_value):
    sigma_img=[]
    energy_map = cv2.imread(img_path)
    for ctu in LCU.CTUs:
        nodes = qd.find_children(ctu.root)
        sigmaCtu = []
        cv2.rectangle(energy_map,
                  (ctu.ctu_x0,ctu.ctu_y0),
                  (ctu.ctu_x0+ctu.ctu_w-1,ctu.ctu_y0+ctu.ctu_h-1),
                  color=(0,0,0),
                  thickness = 1
                  )
        for node in nodes:
            sigma_node = Optimizer_Laplace.compute_sigma_AC(node, LCU.img)
            sigmaCtu.append(sigma_node)
            cv2.rectangle(energy_map,
                          (node.x0,node.y0),
                          (node.x0+node.width-4,node.y0+node.height-4),
                          color=bgr(min_value,max_value,sigma_node),
                          thickness = 1
                          )
        sigma_img.append(sigmaCtu)
    return energy_map

def plotQuantizationMap(img_path,LCU,min_value,max_value,Qi):
    Quantization_map = cv2.imread(img_path)
    i = 0
    for ctu in LCU.CTUs:
        nodes = qd.find_children(ctu.root)
        cv2.rectangle(Quantization_map,
                  (ctu.ctu_x0,ctu.ctu_y0),
                  (ctu.ctu_x0+ctu.ctu_w-1,ctu.ctu_y0+ctu.ctu_h-1),
                  color=(0,0,0),
                  thickness = 1
                  )
        j=0
        for node in nodes:
            cv2.rectangle(Quantization_map,
                          (node.x0,node.y0),
                          (node.x0+node.width-4,node.y0+node.height-4),
                          color=bgr(min_value,max_value,Qi[i][j]),
                          thickness = 1
                          ) 
            j = j+1
        i = i+1
    return Quantization_map

def QuantizationLevelHeatMap(LCU,Qi):
    heat_map = np.zeros((LCU.h,LCU.w))
    i = 0
    for ctu in LCU.CTUs:
        nodes = qd.find_children(ctu.root)
        j=0
        for node in nodes:
            heat_map[node.y0:node.y0+node.height,node.x0:node.x0+node.width] = Qi[i][j]
            j = j+1
        i = i+1
    return heat_map
def EngergyHeatMap(LCU):
    heat_map = np.zeros((LCU.h,LCU.w))
    for ctu in LCU.CTUs:
        nodes = qd.find_children(ctu.root)
        for node in nodes:
            sigma_node = Optimizer_Laplace.compute_sigma_AC(node, LCU.img)
            heat_map[node.y0:node.y0+node.height,node.x0:node.x0+node.width] = sigma_node
    return heat_map

#lam1_min,lam2_min,lam1_max,lam2_max,delta_error_lam1,delta_error_lam2,mu1,mu2,n0,LCU,Rt,Dm
#sigma,mu,E,lam
Test_CTU_Size = [16]
Test_threshold_value = [180]
img_path = "lena512color.tiff"
for CTU_size in Test_CTU_Size:
    for threshold in Test_threshold_value:
        (LCU,img) = qd.ComputeQuadTreeYChannel(img_path,threshold=threshold,minPixelSize=8,CTU_size_h=512,CTU_size_w=512)
        Oj = Optimizer_Laplace.Optimizer(lam1_min = 0,lam2_min=0,mu1=0.1,mu2=1,n0=0.1,Qtilemax=7,LCU=LCU,Rt=.2,Dm=1000,delta_Rate_Target=0.1)
        # Oj.plot_RD_curve(0.1,0.01,0.02,0.05,.5,63,0)
        # DisplayEnergyMap(plotEnergymap(img_path,LCU,0,15),"Threshold %s CTU Size %s" %(CTU_size,threshold))
        
        heat_map_energy = EngergyHeatMap(LCU)
        plt.title("Energy heatmap CTU size %s threshold %s" %(CTU_size,threshold))
        plt.imshow(heat_map_energy, cmap='viridis' )
        plt.colorbar(anchor=(0.5,1),aspect=5)
        plt.show()
        (Q1,Q2,D1_est,D2_est,R1_est,R2_est) = Oj.optimize_LCU(Optimizer_Laplace.compute_sigma_AC)
        
        DisplayResult(LCU,img,"Threshold %s CTU Size %s D1 estimated %s D2 estimated %s R1 estimated %s R2 estimated %s " %(CTU_size,threshold,D1_est,D2_est,R1_est,R2_est))
        
        # DisplayQuantizationMap(plotQuantizationMap(img_path,LCU,0,15,Q1),
        #             plotQuantizationMap(img_path,LCU,0,15,Q2),"Threshold %s CTU Size %s" %(CTU_size,threshold))
        

        
        heat_map_Q1 = QuantizationLevelHeatMap(LCU,Q1)
        heat_map_Q2 = QuantizationLevelHeatMap(LCU,Q2)
        plt.figure(4,figsize=(20,20))
        plt.subplot(1,2,1)
        plt.imshow(heat_map_Q1, cmap='viridis' )
        plt.title("Quantization D1 heatmap CTU size %s threshold %s" %(CTU_size,threshold))

        plt.colorbar(anchor=(0.5,1),aspect=5)
        plt.subplot(1,2,2)

        plt.imshow(heat_map_Q2, cmap='viridis' )
        plt.title("Quantization D2 heatmap CTU size %s threshold %s" %(CTU_size,threshold))

        plt.colorbar(anchor=(0.5,1),aspect=5)
        plt.show()

import pickle
with open('LCU.pkl', 'wb') as output:
    pickle.dump(Oj,output, pickle.HIGHEST_PROTOCOL)