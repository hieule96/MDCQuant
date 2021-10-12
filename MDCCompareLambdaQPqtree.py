#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 21:19:54 2021

@author: ubuntu
"""

import subprocess
import YUVLib
import Optimizer
import Quadtreelib as qd
import re
import numpy as np
import Optimizer as Opt
import sys
import skimage.metrics
import csv
import signal
from functools import partial
import pdb
import skimage.metrics
WorkDir = "outputs/"
ConfigDir = "MDC_cfg/"
fields = ['QP','lambda','PSNR0','R0(Mbits)','PSNR1','R1(Mbits)','PSNR2','R2(Mbits)']

class MDC_DEC_STATE:
    SEEK_FRAME = 0
    FRAME_PROCESSING = 1
    FRAME_WRITE = 2
def writenpArrayToFile(YUVArray,outputFileName,mode='wb'):
    with open(outputFileName,mode) as output:
        output.write(YUVArray[0].tobytes())
        output.write(YUVArray[1].tobytes())
        output.write(YUVArray[2].tobytes())
class EncoderConfigFile():
    def __init__(self,qTreeFileName,yuvResiFileName,q1FileName,q2FileName,yuvOrgFileName,q0FileName=""):
        self.qTreeFileName=qTreeFileName
        self.yuvResiFileName=yuvResiFileName
        self.q1FileName=q1FileName
        self.q2FileName=q2FileName
        self.q0FileName=q0FileName
        self.yuvOrgFileName=yuvOrgFileName
class PictureParamter():
    def __init__(self,w,h,nbCUinCTU,frameToEncode):
        self.bord_h = h
        self.bord_w = w
        self.nbCUinCTU=nbCUinCTU
        self.frameToEncode = frameToEncode
        self.step_w = np.ceil(w/64)
        self.step_h = np.ceil(h/64)
class DecoderConfigFile():
    def __init__(self,qTreeFileName,yuvOrgFileName,reconD1FileName,reconD2FileName,q1FileName,q2FileName):
        self.qTreeFileName = qTreeFileName
        self.reconD1FileName = reconD1FileName
        self.reconD2FileName = reconD2FileName
        self.yuvOrgFileName = yuvOrgFileName
        self.q1FileName = q1FileName
        self.q2FileName = q2FileName
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
                                QPFile1.write("%d," %(Q1[i]))
                                QPFile2.write("%d," %(Q2[i]))
                                QPFile0.write("%d," %(Q0[i]))
                            QPFile1.write("\n") 
                            QPFile2.write("\n") 
                            QPFile0.write("\n")
def loadQtreeStructure(configFile,pictureParameter,frame_begin,frame_end):
    lcu_list=[]
    with open(configFile.qTreeFileName,'r') as file:
        for lines in file:
            ParseTxt = lines
            matchObj  = re.sub('[<>]',"",ParseTxt)      
            matchObj  = re.sub('[ ]',",",matchObj)      
            chunk = matchObj.split(',')
            frame = int(chunk[0])
            pos = int(chunk[1])
            if (frame>=frame_begin and frame<frame_end):
                if pos == 0:
                    lcu = 0
                    imgY= YUVLib.read_YUV420_frame(open(configFile.yuvResiFileName,"rb"),pictureParameter.bord_w,pictureParameter.bord_h,frame)._Y
                    lcu = qd.LargestCodingUnit(imgY.astype(np.float32) - 128,1,8)
                    step_w = int (np.around(lcu.w / lcu.block_size_w))
                quadtree_composition = chunk[2:]
                CTU = qd.Node(int (pos%step_w)*64,int (pos/step_w)*64,64,64)
                qd.import_subdivide(CTU,quadtree_composition,0)
                lcu.CTUs.append(CTU)
                lcu.nbCTU = lcu.nbCTU + 1
                if pos == pictureParameter.nbCUinCTU-1:
                    lcu.convert_Tree_childrenArray()
                    lcu.remove_bord_elements(lcu.w,lcu.h)
                    lcu.Init_aki()
                    lcu.merge_CTU()
                    lcu_list.append(lcu)
                    Opt.Optimizer_curvefitting.initCoefficient(lcu)
    return lcu_list
def processLCU(lcu,MDC_param,frame):
    MDC_param.LCU = lcu
    Oj = Opt.Optimizer_curvefitting(MDC_param)
    (Q1,Q2,D1_est,D2_est,R1_est,R2_est) = Oj.optimize_LCU() 
    Q1 = np.array(Q1,dtype=np.uint8)
    Q1 = Q1.ravel()
    Q2 = np.array(Q2,dtype=np.uint8)
    Q2 = Q2.ravel()
    Q0 = np.minimum(Q1,Q2)
    return [[frame,lcu.nbCUperCTU,[Q0,Q1,Q2]]]
def processFrame(frame_begin,frame_end,MDC_param,configFile,pictureParameter):
    frame = 0
    Q0 = []
    Q1 = []
    Q2 = []
    lcu = []
    output_list = []
    with open(configFile.qTreeFileName,'r') as file:
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
                    imgY= YUVLib.read_YUV420_frame(open(configFile.yuvResiFileName,"rb"),pictureParameter.bord_w,pictureParameter.bord_h,frame)._Y
                    lcu = qd.LargestCodingUnit(imgY.astype(np.float32) - 128,1,8)
                    step_w = int (np.around(lcu.w / lcu.block_size_w))
                quadtree_composition = chunk[2:]
                CTU = qd.Node(int (pos%step_w)*64,int (pos/step_w)*64,64,64)
                qd.import_subdivide(CTU,quadtree_composition,0)
                lcu.CTUs.append(CTU)
                lcu.nbCTU = lcu.nbCTU + 1
                if pos == pictureParameter.nbCUinCTU-1:
                    lcu.convert_Tree_childrenArray()
                    lcu.remove_bord_elements(lcu.w,lcu.h)
                    lcu.Init_aki()
                    lcu.merge_CTU()
                    # png.from_array(lcu.render_img(imgY,thickness=1,color=(255,255,255)),'L').save(outputImage_PATH+"Qtree_frame%s.png" %(frame))
                    Opt.Optimizer_curvefitting.initCoefficient(lcu)
                    MDC_param.LCU = lcu
                    Oj = Opt.Optimizer_curvefitting(MDC_param)
                    (Q1,Q2,D1_est,D2_est,R1_est,R2_est) = Oj.optimize_LCU() 
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
def assignBlock(x,y,blockSize,YUVJoint,Frame):
    x_half = x//2
    y_half = y//2
    blockSize_half = blockSize//2
    YUVJoint[0][y:y+blockSize,x:x+blockSize] = Frame._Y[y:y+blockSize,x:x+blockSize]
    YUVJoint[2][y_half:y_half+blockSize_half,x_half:x_half+blockSize_half] = Frame._U[y_half:y_half+blockSize_half,x_half:x_half+blockSize_half] 
    YUVJoint[1][y_half:y_half+blockSize_half,x_half:x_half+blockSize_half] = Frame._V[y_half:y_half+blockSize_half,x_half:x_half+blockSize_half]
def checkDecision(x,y,blockSize,imgO,img1,img2):
    decision_MSE = 0
    mseblock1 = skimage.metrics.mean_squared_error(imgO._Y[y:y+blockSize,x:x+blockSize],img1._Y[y:y+blockSize,x:x+blockSize])
    mseblock2 = skimage.metrics.mean_squared_error(imgO._Y[y:y+blockSize,x:x+blockSize],img2._Y[y:y+blockSize,x:x+blockSize])
    if mseblock1>mseblock2:
        decision_MSE = 2
    elif mseblock1<mseblock2:
        decision_MSE = 1
    return decision_MSE
def decodeFrame(configFile,pictureParam):
    dec_state = MDC_DEC_STATE.SEEK_FRAME
    frame = 0
    PNSR_mean = 0
    P1 = 0
    P2 = 0
    P0 = 0
    with open(configFile.qTreeFileName,'r') as qtFile:
        with open(configFile.q1FileName,'r') as quant1File:
            with open(configFile.q2FileName,'r') as quant2File:
                while (True):
                    if (dec_state == MDC_DEC_STATE.SEEK_FRAME):
                        img1 = YUVLib.read_YUV420_frame(open(configFile.reconD1FileName,"rb"),pictureParam.bord_w,pictureParam.bord_h,frame)
                        img2 = YUVLib.read_YUV420_frame(open(configFile.reconD2FileName,"rb"),pictureParam.bord_w,pictureParam.bord_h,frame)
                        imgO = YUVLib.read_YUV420_frame(open(configFile.yuvOrgFileName,"rb"),pictureParam.bord_w,pictureParam.bord_h,frame)
                        YUVJoint = [[],[],[]]
                        YUVJoint[0] = np.zeros((img1._Y.shape[0],img1._Y.shape[1]),dtype=np.uint8)
                        YUVJoint[1] = np.zeros((img1._V.shape[0],img1._V.shape[1]),dtype=np.uint8)
                        YUVJoint[2] = np.zeros((img1._U.shape[0],img1._U.shape[1]),dtype=np.uint8)
                        dec_state = MDC_DEC_STATE.FRAME_PROCESSING
                        print ("SEEK YUV ",(frame))
                    elif (dec_state == MDC_DEC_STATE.FRAME_PROCESSING):
                        for lines in qtFile:
                            ParseTxt = lines
                            matchObj  = re.sub('[<>]',"",ParseTxt)      
                            matchObj  = re.sub('[ ]',",",matchObj)      
                            chunk = matchObj.split(',')
                            position_cu = int(chunk[1])
                            quadtree_composition = chunk[2:]
                            CTU = qd.Node(int (position_cu%pictureParam.step_w)*64,int (position_cu/pictureParam.step_w)*64,64,64)
                            qd.import_subdivide(CTU,quadtree_composition,0)
                            
                            # Cbf1 = cbf1File.readline().split(",")
                            # Cbf1 = [int(i) for i in Cbf1[:-1]]
                            # Cbf2 = cbf2File.readline().split(",")
                            # Cbf2 = [int(i) for i in Cbf2[:-1]]
                            # pdb.set_trace()
                            Q1 = quant1File.readline().split(",")
                            Q2 = quant2File.readline().split(",")
                            cus = qd.find_children(CTU)
                            #Remove bord elements
                            remove_list = []
                            i = 0
                            for cu in cus:
                                if(cu.x0 > pictureParam.bord_w or cu.y0>pictureParam.bord_h or cu.x0+cu.width > pictureParam.bord_w or cu.y0+cu.height > pictureParam.bord_h):
                                    remove_list.append(i)
                                i = i + 1
                            i = 0
                            for pos in remove_list:
                                # print (ctu_index,remove_list)
                                cus.pop(pos-i)
                                i = i + 1
                            for index in range (0,len(cus)):
                                decision = 0
                                if Q1[index] < Q2[index]:
                                    decision = 1
                                elif Q1[index] > Q2[index]:
                                    decision = 2
                                else:
                                    #else check the block of references
                                    if (Q1[0] < Q2[0]):
                                        decision = 1
                                    elif (Q1[0] > Q2[0]):
                                        decision = 2
                                #check decision based on QP value is valid ?
                                testdecision = checkDecision(cus[index].x0,cus[index].y0,cus[index].height,imgO,img1,img2)
                                if (testdecision!=decision and testdecision!=0):
                                    # print (testdecision,decision)
                                    # print("Bad decision based on QP")
                                    decision=testdecision
                                if decision == 1:
                                    assignBlock(cus[index].x0,cus[index].y0,cus[index].height,YUVJoint,img1)
                                elif decision == 2:
                                    assignBlock(cus[index].x0,cus[index].y0,cus[index].height,YUVJoint,img2)
                                elif decision == 0:
                                    assignBlock(cus[index].x0,cus[index].y0,cus[index].height,YUVJoint,img2)
                            if (position_cu >= pictureParam.nbCUinCTU-1):
                                dec_state = MDC_DEC_STATE.FRAME_WRITE
                                break
                    elif (dec_state == MDC_DEC_STATE.FRAME_WRITE):
                        P1 = skimage.metrics.peak_signal_noise_ratio(imgO._Y,img1._Y)
                        P2 = skimage.metrics.peak_signal_noise_ratio(imgO._Y,img2._Y)
                        P0 = skimage.metrics.peak_signal_noise_ratio(imgO._Y,YUVJoint[0])
                        print ("WRITE_FRAME %s PNSR1: %s PSNR2: %s PSNR0: %s" %(frame,P1,P2,P0))
                        PNSR_mean = P0 + PNSR_mean 
                        # rgbD0 = convertYUVToBGR(YUVJoint[0],YUVJoint[2],YUVJoint[1])
                        # rgbD1 = convertYUVToBGR(img1._Y,img1._V,img1._U)
                        # rgbD2 = convertYUVToBGR(img2._Y,img2._V,img2._U)
    
                        # rgbAll = np.hstack((rgbD0,rgbD1,rgbD2))
                        # cv2.imshow("D0 - D1 - D2",rgbAll)
                        # cv2.waitKey(0)
    
                        YUVJoint[0] = YUVJoint[0].ravel()
                        YUVJoint[1] = YUVJoint[1].ravel()
                        YUVJoint[2] = YUVJoint[2].ravel()
    
                        if (frame == 0):
                            writenpArrayToFile(YUVJoint,"rec_D0.yuv",'wb')
                        else:
                            writenpArrayToFile(YUVJoint,"rec_D0.yuv",'ab')
                        dec_state = MDC_DEC_STATE.SEEK_FRAME
                        frame = frame + 1
                    if (frame >= pictureParam.frameToEncode):
                        print ("BREAK")
                        PSNR_mean = PNSR_mean/pictureParam.frameToEncode
                        print ("PNSR0 mean",PSNR_mean)
                        # cv2.destroyAllWindows()
                        break
    return P0,P1,P2
def run_encoder_MDC(qtreeFileName,MDCindex,qpmean):
    process = subprocess.Popen(["./TAppEncoder","-c" ,ConfigDir+"encoder_intra_main-D%d.cfg"%(MDCindex),"-c","news_cif.cfg","-qt",qtreeFileName,"-q","%d" %(qpmean)],stdout=subprocess.PIPE)
    bitrate = 0
    while True:
      output = process.stdout.readline()
      if process.poll() is not None:
        break
      if output:
        outputparse = output.strip().decode("utf-8")
        parse_result = re.match("POC *\s *(-?[0-9]+) .*\  *(-?[0-9]+) bits",outputparse)
        if (parse_result != None):
            bitrate = parse_result.group(parse_result.lastindex)
    return bitrate
def generateQtreeEncoderPSNRMSE(qp):
    process = subprocess.Popen(["./TAppEncoder","-c" ,"encoder_intra_main.cfg","-c","news_cif.cfg","-qt", "outputs/qtree%d.txt" %(qp),"-resi","outputs/resiQP%d.yuv" %(qp), "-q"," %d"%(qp)],stdout=subprocess.PIPE)
    bitrate = 0
    psnr_Y = 0
    while True:
      output = process.stdout.readline()
      if process.poll() is not None:
        break
      if output:
        outputparse = output.strip().decode("utf-8")
        parse_result = re.match("POC *\s (-?[0-9]+) .*\s *(-?[0-9]+) bits \[Y *(-?[0-9.]+) dB *\s U *(-?[0-9.]+) dB *\s V (-?[0-9.]+) dB\] \[Y MSE *(-?[0-9.]+) *\s U MSE *(-?[0-9.]+) *\s V MSE *\s *(-?[0-9.]+)\]",outputparse)
        if (parse_result != None):
            bitrate = parse_result.group(2) 
            psnr_Y = parse_result.group(3) 
            mse_Y = parse_result.group(6)
    return bitrate,psnr_Y,mse_Y

def exit_correcly_save(signum,frame,StatList):
    signal.signal (signal.SIGINT, original_sigint)
    try:
        print("Saving progress...")
        with open("Stat.csv","w") as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            writer.writerows(Stat)
        print ("Done")
        sys.exit(0)
    except KeyboardInterrupt:
        print("Ok ok, quitting")
        sys.exit(1)

    # restore the exit gracefully handler here    
    signal.signal(signal.SIGINT, exit_correcly_save)
    
pictureParam=PictureParamter(352,288,30,1)
encoderConfigFile = EncoderConfigFile(qTreeFileName="",yuvResiFileName="",q1FileName="QP1.csv",q2FileName="QP2.csv",yuvOrgFileName="news_cif.yuv",q0FileName="QP0.csv")
decoderConfigFile = DecoderConfigFile(qTreeFileName="", yuvOrgFileName ="news_cif.yuv",reconD1FileName="rec_D1.yuv", reconD2FileName="rec_D2.yuv", q1FileName="QP1.csv", q2FileName="QP2.csv")

# StatEncoder = []
# fields=["QP","Bits(Mbits)","PSNRY","MSEY"]
# for qp in range (0,51):
#     bitrate,psnr_Y,mse_Y=generateQtreeEncoderPSNRMSE(qp)
#     StatEncoder.append([qp,int (bitrate)*1e-6,psnr_Y,mse_Y])
    
# with open("Stat_Encoder.csv","w") as file:
#     writer = csv.writer(file)
#     writer.writerow(fields)
#     writer.writerows(StatEncoder)

Stat = []
original_sigint = signal.signal(signal.SIGINT, partial(exit_correcly_save,Stat))
for i in range (0,50,2):
    encoderConfigFile.qTreeFileName = WorkDir+"qtree%d.txt"%(i)
    encoderConfigFile.yuvResiFileName = WorkDir+"resiQP%d.yuv"%(i)
    lcu = loadQtreeStructure(encoderConfigFile,pictureParam,0,1)
    for lam in np.arange (0.0,100,10):
        print (i,lam)
        MDC_Param = Opt.OptimizerParameterLambdaCst(lam1=lam,lam2=lam,mu1=0.1,mu2=0.1,n0=0.5,QPmax=51,LCU=[],Dm=2000000,rN=0.0)
        result = []
        result.append(processLCU(lcu[0],MDC_Param,0))
        writeToFile(encoderConfigFile.q0FileName,encoderConfigFile.q1FileName,encoderConfigFile.q2FileName,"w",result)
        QP1_mean = result[0][0][2][1].mean()
        QP2_mean = result[0][0][2][2].mean()
        bitrateD1 = run_encoder_MDC(encoderConfigFile.qTreeFileName,1,np.int16(QP1_mean))
        bitrateD2 = run_encoder_MDC(encoderConfigFile.qTreeFileName,2,np.int16(QP2_mean))
        decoderConfigFile.qTreeFileName = encoderConfigFile.qTreeFileName
        P0,P1,P2 = decodeFrame(decoderConfigFile,pictureParam)
        Stat.append([i,lam,P0,(int (bitrateD1)+int(bitrateD2))*1e-6,P1,int (bitrateD1)*1e-6,P2,int(bitrateD2)*1e-6])
        print (bitrateD1,bitrateD2)
        print (P0,P1,P2)
with open("Stat.csv","w") as file:
    writer = csv.writer(file)
    writer.writerow(fields)
    writer.writerows(Stat)
