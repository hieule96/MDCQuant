# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 16:18:40 2021

@author: LE Trung Hieu
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 23:54:54 2021

@author: LE Trung Hieu
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import Quadtreelib as qd
import Optimizer as Opt
import png
import re
import os
import time
import multiprocessing as mp
import skimage.metrics
import pdb

frame = 0
PSNR0_seq=[]
PSNR1_seq=[]
PSNR2_seq=[]
R0_seq = []
R0_AC_seq = []
Rt = 1.0


bord_h = 288
bord_w = 352
step_w = np.ceil (bord_w/64)
step_h = np.ceil (bord_h/64)
nbCUinCTU = 30
nbframeToEncode = 30
step_spliting = 1

class FrameYUV(object):
    def __init__(self, Y, U, V):
        self._Y = Y
        self._U = U
        self._V = V

def read_YUV420_frame(fid, width, height,frame=0):
    # read a frame from a YUV420-formatted sequence
    d00 = height // 2
    d01 = width // 2
    fid.seek(frame*(width*height+width*height//2))
    Y_buf = fid.read(width * height)
    Y = np.reshape(np.frombuffer(Y_buf, dtype=np.uint8), [height, width])
    U_buf = fid.read(d01 * d00)
    U = np.reshape(np.frombuffer(U_buf, dtype=np.uint8), [d00, d01])
    V_buf = fid.read(d01 * d00)
    V = np.reshape(np.frombuffer(V_buf, dtype=np.uint8), [d00, d01])
    return FrameYUV(Y, U, V)

def DisplayResultandExportPNG(LCU,img,path,sequence_number):
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
    newimgD2 = newimgD2.astype(np.uint8)    
    
    png.from_array(newimgDC,'L').save(path + "central_seq%s.png" %(sequence_number))
    png.from_array(newimgD1,'L').save(path + "D1_seq%s.png" %(sequence_number))
    png.from_array(newimgD2,'L').save(path + "D2_seq%s.png" %(sequence_number))

    R1_AC = quant.LCU_CalRateQt(LCU,imgdctD1,AC_Remove = True)
    R2_AC = quant.LCU_CalRateQt(LCU,imgdctD2,AC_Remove = True)
    R1_DC = R2_DC = quant.LCU_CalRateQt_DC_DPCM(LCU,imgdctD1)
    R1 = R1_AC + R1_DC
    R2 = R2_AC + R2_DC
    D0 = np.around (mean_squared_error(newimgDC,img[:LCU.h,:LCU.w]),3)
    D1 = np.around (mean_squared_error(newimgD1,img[:LCU.h,:LCU.w]),3)
    D2 = np.around (mean_squared_error(newimgD2,img[:LCU.h,:LCU.w]),3)
    PSNR0 = np.around (skimage.metrics.peak_signal_noise_ratio(img[:LCU.h,:LCU.w],newimgDC,data_range=255),3)
    PSNR1 = np.around (skimage.metrics.peak_signal_noise_ratio(img[:LCU.h,:LCU.w],newimgD1,data_range=255),3)
    PSNR2 = np.around (skimage.metrics.peak_signal_noise_ratio(img[:LCU.h,:LCU.w],newimgD2,data_range=255),3)
    
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
    plt.subplot(321), plt.imshow(newimgD1,cmap='gray'), plt.title("D1 MSE: %s PNSR: %s dB R1_AC: %s bpp R1_T %s bpp" %(D1,PSNR1,R1_AC,R1))
    plt.subplot(322), plt.imshow(newimgD2,cmap='gray'), plt.title("D2 MSE: %s PNSR: %s dB R2_AC: %s bpp R2_T %s bpp" %(D2,PSNR2,R2_AC,R2))
    plt.subplot(323), plt.imshow(newimgDC,cmap='gray'), plt.title("D0 MSE: %s PNSR: %s dB R0_AC: %s bpp R0_T %s bpp" %(D0,PSNR0,R0_AC,R0))
    plt.show()
    return PSNR0,PSNR1,PSNR2,R0,R0_AC

CTU_path = "decoder_cupu.txt"
yuv_files = "news_cif.yuv"
outputVideo_PATH = "outputs/videos/"
outputImage_PATH = "outputs/images/"
outputQP_PATH = ""
Q1FileName = "QP1.csv"
Q2FileName = "QP2.csv"
Q0FileName = "QP0.csv"
if os.path.isdir(outputVideo_PATH) == False:
    os.makedirs(outputVideo_PATH)
if os.path.isdir(outputImage_PATH) == False:
    os.makedirs(outputImage_PATH)



def processFramePlot(frame_begin,frame_end,mu1=0.5,mu2=0.5,n0=0.5,Dm=200):
    frame = 0
    lcu = []
    output_list = []
    PNSR0_list = []
    PNSR1_list = []
    PNSR2_list = []
    R0_list = []
    with open(CTU_path,'r') as file:
        for lines in file:
            ParseTxt = lines
            matchObj  = re.sub('[<>]',"",ParseTxt)      
            matchObj  = re.sub('[ ]',",",matchObj)      
            chunk = matchObj.split(',')
            frame = int(chunk[0])
            pos = int(chunk[1])
            #print (lines)
            if (frame>=frame_begin and frame<frame_end):
                if pos == 0:
                    lcu = 0
                    imgY= read_YUV420_frame(open(yuv_files,"rb"),bord_w,bord_h,frame)._Y
                    lcu = qd.LargestCodingUnit(imgY.astype(np.float32) - 128,1,8)
                    step_w = int (np.around(lcu.w / lcu.block_size_w))
                quadtree_composition = chunk[2:]
                CTU = qd.Node(int (pos%step_w)*64,int (pos/step_w)*64,64,64)
                qd.import_subdivide(CTU,quadtree_composition,0)
                lcu.CTUs.append(CTU)
                lcu.nbCTU = lcu.nbCTU + 1

        
                if pos == nbCUinCTU-1:
                    lcu.convert_Tree_childrenArray()
                    lcu.remove_bord_elements(lcu.w,lcu.h)
                    lcu.Init_aki()
                    lcu.merge_CTU()
                    png.from_array(lcu.render_img(imgY,thickness=1,color=(255,255,255)),'L').save(outputImage_PATH+"Qtree_frame%s.png" %(frame))
                    Opt.Optimizer_curvefitting.initCoefficient(lcu)
                    # LCU.ExportParamtertoCSV(img_path)
                    for i in range (5,100,5):
                        GlobalParam = Opt.OptimizerParameterLambdaCst(lam1=i,lam2=i,mu1=mu1,mu2=mu2,n0=0.5,LCU=lcu,Dm=Dm)
                        Oj = Opt.Optimizer_curvefitting(GlobalParam)
                        (Q1,Q2,D1_est,D2_est,R1_est,R2_est) = Oj.optimize_LCU() 
                        PSNR0,PSNR1,PSNR2,R0,R0_AC = DisplayResultandExportPNG(lcu,imgY,outputImage_PATH,frame)
                        PNSR0_list.append(PSNR0)
                        PNSR1_list.append(PSNR1)
                        PNSR2_list.append(PSNR2)
                        R0_list.append(R0)
                    plt.plot(R0_list,PNSR0_list),plt.title("PSNR0")
                    plt.plot(R0_list,PNSR1_list),plt.title("PSNR1")
                    plt.plot(R0_list,PNSR2_list),plt.title("PSNR2")
                    plt.show()
            if frame == frame_end:
                break
        return output_list  
    
if __name__=='__main__':
    processFramePlot(0,1)

