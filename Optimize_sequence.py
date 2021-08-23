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
    newimgD2 = newimgD1.astype(np.uint8)    
    
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

def processFrame(frame_begin,frame_end):
    frame = 0
    Q0 = []
    Q1 = []
    Q2 = []
    lcu = []
    output_list = []
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
                    GlobalParam = Opt.OptimizerParameterLambdaCst(lam1=100,lam2=100,mu1=0.1,mu2=0.1,n0=0.5,QPmax=51,LCU=lcu,Dm=200)
                    Oj = Opt.Optimizer_curvefitting(GlobalParam)
                    (Q1,Q2,D1_est,D2_est,R1_est,R2_est) = Oj.optimize_LCU() 
                    PSNR0,PSNR1,PSNR2,R0,R0_AC = DisplayResultandExportPNG(lcu,imgY,outputImage_PATH,frame)
                    PSNR0_seq.append(PSNR0)
                    PSNR1_seq.append(PSNR1)
                    PSNR2_seq.append(PSNR2)
                    R0_seq.append(R0)
                    R0_AC_seq.append(R0_AC)
                    Q1 = np.array(Q1,dtype=np.uint8)
                    Q1 = Q1.ravel()
                    Q2 = np.array(Q2,dtype=np.uint8)
                    Q2 = Q2.ravel()
                    Q0 = np.minimum(Q1,Q2)
                    output_list.append([frame,lcu.nbCUperCTU,[Q0,Q1,Q2]])
                    print (frame)
            if frame == frame_end:
                break
        return output_list  
    
def writeToFile(Q0FileName,Q1FileName,Q2FileName,mode,result):
    with open(Q1FileName,mode) as QPFile1:
        with open(Q2FileName,mode) as QPFile2:
            with open(Q0FileName,mode) as QPFile0:
                for framePackedGroupFrame in result:
                    for framePackedResult in framePackedGroupFrame:
                        nbCUperCTU = framePackedResult[1]
                        Q1 = framePackedResult[2][1]
                        Q2 = framePackedResult[2][2]
                        Q0 = framePackedResult[2][0]
                        for nbCTU in nbCUperCTU:
                            for i in range (nbCTU):
                                QPFile1.write("%d " %(Q1[i]))
                                QPFile2.write("%d " %(Q2[i]))
                                QPFile0.write("%d " %(Q0[i]))
                            QPFile1.write("\n") 
                            QPFile2.write("\n") 
                            QPFile0.write("\n")
if __name__=='__main__':
    process_time_begin = time.time()

    # p1 = mp.Process(target = processFrame, args = (0,10,"temp1.csv","temp2.csv","temp3.csv","decoder_cupu_fixed.txt"))
    # p2 = mp.Process(target = processFrame, args = (10,20,"temp4.csv","temp5.csv","temp6.csv","decoder_cupu_fixed.txt"))
    # p3 = mp.Process(target = processFrame, args = (10,20,"temp4.csv","temp5.csv","temp6.csv","decoder_cupu_fixed.txt"))
    # p4 = mp.Process(target = processFrame, args = (40,50,"temp10.csv","temp11.csv","temp12.csv","decoder_cupu_fixed.txt"))

    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    
    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()
    # result = []
    # result.append(processFrame(961, 962))
    # writeToFile(Q0FileName,Q1FileName,Q2FileName,"w",result)    
    processFrame(0,1)
    # splitting a sequence into multiples 10 frame each
    # segment = []
    # for i in range (1,nbframeToEncode,step_spliting):
    #     segment.append([i,i+step_spliting])
    
    # firstWrite = True
    # data = tuple(segment)
    # with mp.Pool() as p:
    #     result = p.starmap(processFrame,data)
    # if (firstWrite==True):
    #     writeToFile(Q0FileName,Q1FileName,Q2FileName,"w",result)
    #     firstWrite = False
    # else:
    #     writeToFile(Q0FileName,Q1FileName,Q2FileName,"a",result)
    # print (result)
    # print ("Time process First 100:", time.time() - process_time_begin)


# try:
#     (ffmpeg
#     .input(outputImage_PATH+'/central_seq%d.png',framerate=10)
#     .output(outputVideo_PATH+'/central_seq.avi')
#     .overwrite_output()
#     .run(capture_stdout=True, capture_stderr=True))
# except ffmpeg.Error as e:
#     print('stdout:', e.stdout.decode('utf8'))
#     print('stderr:', e.stderr.decode('utf8'))
#     raise e

# try:
#     (ffmpeg
#     .input(outputImage_PATH+'/D1_seq%d.png',framerate=10)
#     .output(outputVideo_PATH+'/D1_seq.avi')
#     .overwrite_output()
#     .run(capture_stdout=True, capture_stderr=True))
# except ffmpeg.Error as e:
#     print('stdout:', e.stdout.decode('utf8'))
#     print('stderr:', e.stderr.decode('utf8'))
#     raise e

# try:
#     (ffmpeg
#     .input(outputImage_PATH+'/D2_seq%d.png',framerate=10)
#     .output(outputVideo_PATH+'/D2_seq.avi')
#     .overwrite_output()
#     .run(capture_stdout=True, capture_stderr=True))
# except ffmpeg.Error as e:
#     print('stdout:', e.stdout.decode('utf8'))
#     print('stderr:', e.stderr.decode('utf8'))
#     raise e
# try:
#     (ffmpeg
#     .input(outputImage_PATH+'/plotcomapre%d.png',framerate=10)
#     .output(outputVideo_PATH+'/plotcomapre.avi')
#     .overwrite_output()
#     .run(capture_stdout=True, capture_stderr=True))
# except ffmpeg.Error as e:
#     print('stdout:', e.stdout.decode('utf8'))
#     print('stderr:', e.stderr.decode('utf8'))
#     raise e
# with open("perf.csv",'w') as file:
#     file.write("PSNR0,PSNR1,PSNR2,R0,R0_AC\n")
#     for PSNR0,PSNR1,PSNR2,R0,R0_AC in zip(PSNR0_seq,PSNR1_seq,PSNR2_seq,R0_seq,R0_AC_seq):
#         file.write("%s,%s,%s,%s,%s\n"%(PSNR0,PSNR1,PSNR2,R0,R0_AC))
# file.close()