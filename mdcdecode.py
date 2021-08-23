# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:02:38 2021

@author: LE Trung Hieu
"""
import cv2
import numpy as np
import Quadtreelib as qd
import re
import skimage.metrics

DecQ1FileName = "DecQP1.csv"
DecQ2FileName = "DecQP2.csv"
DecQtreeFile  = "QtreeOut.txt"
yuvD1_files = "reconstrutedD1.yuv"
yuvD2_files = "reconstrutedD2.yuv"
yuvD0_file = "reconstructedD0.yuv"
yuvO = "news_cif.yuv"
step_w = np.ceil (352/64)
step_h = np.ceil (288/64)
bord_h = 288
bord_w = 352
nbCUinCTU = 30
nbFrameToDecode = 30
frame = 0


class FrameYUV(object):
    def __init__(self, Y, U, V):
        self._Y = Y
        self._U = U
        self._V = V
class MDC_DEC_STATE:
    SEEK_FRAME = 0
    FRAME_PROCESSING = 1
    FRAME_WRITE = 2
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
def convertYUVToBGR(y,v,u):
    u = cv2.resize(u,(y.shape[1],y.shape[0]))
    v = cv2.resize(v,(y.shape[1],y.shape[0]))
    yvu = cv2.merge((y, v, u))
    bgr = cv2.cvtColor(yvu, cv2.COLOR_YCrCb2BGR);
    return bgr
def writenpArrayToFile(YUVArray,outputFileName,mode='wb'):
    with open(outputFileName,mode) as output:
        output.write(YUVArray[0].tobytes())
        output.write(YUVArray[1].tobytes())
        output.write(YUVArray[2].tobytes())
dec_state = MDC_DEC_STATE.SEEK_FRAME
with open(DecQtreeFile,'r') as qtFile:
    with open(DecQ1FileName,'r') as quant1File:
        with open(DecQ2FileName,'r') as quant2File:
            while (True):
                if (dec_state == MDC_DEC_STATE.SEEK_FRAME):
                    img1 = read_YUV420_frame(open(yuvD1_files,"rb"),bord_w,bord_h,frame)
                    img2 = read_YUV420_frame(open(yuvD2_files,"rb"),bord_w,bord_h,frame)
                    imgO = read_YUV420_frame(open(yuvO,"rb"),bord_w,bord_h,frame)
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
                        CTU = qd.Node(int (position_cu%step_w)*64,int (position_cu/step_w)*64,64,64)
                        qd.import_subdivide(CTU,quadtree_composition,0)
                        Q1 = quant1File.readline().split(",")
                        Q2 = quant2File.readline().split(",")
                        Q1 = [int(i) for i in Q1[:-1]]
                        Q2 = [int(i) for i in Q2[:-1]]
                        cus = qd.find_children(CTU)
                        #Remove bord elements
                        remove_list = []
                        i = 0
                        for cu in cus:
                            if(cu.x0 > bord_w or cu.y0>bord_h or cu.x0+cu.width > bord_w or cu.y0+cu.height > bord_h):
                                remove_list.append(i)
                            i = i + 1
                        i = 0
                        for pos in remove_list:
                            # print (ctu_index,remove_list)
                            cus.pop(pos-i)
                            i = i + 1
                        for index in range (0,len(cus)):
                            if Q1[index] < Q2[index]:
                                YUVJoint[0][cus[index].y0:cus[index].y0 + cus[index].height,cus[index].x0:cus[index].x0 + cus[index].width] = img1._Y[cus[index].y0:cus[index].y0 + cus[index].height,cus[index].x0:cus[index].x0 + cus[index].width]
                                YUVJoint[1][cus[index].y0//2:cus[index].y0//2 + cus[index].height//2,cus[index].x0//2:cus[index].x0//2 + cus[index].width//2] = img1._U[cus[index].y0//2:cus[index].y0//2 + cus[index].height//2,cus[index].x0//2:cus[index].x0//2 + cus[index].width//2]
                                YUVJoint[2][cus[index].y0//2:cus[index].y0//2 + cus[index].height//2,cus[index].x0//2:cus[index].x0//2 + cus[index].width//2] = img1._V[cus[index].y0//2:cus[index].y0//2 + cus[index].height//2,cus[index].x0//2:cus[index].x0//2 + cus[index].width//2]
                            else:
                                YUVJoint[0][cus[index].y0:cus[index].y0 + cus[index].height,cus[index].x0:cus[index].x0 + cus[index].width] = img2._Y[cus[index].y0:cus[index].y0 + cus[index].height,cus[index].x0:cus[index].x0 + cus[index].width]
                                YUVJoint[1][cus[index].y0//2:cus[index].y0//2 + cus[index].height//2,cus[index].x0//2:cus[index].x0//2 + cus[index].width//2] = img2._U[cus[index].y0//2:cus[index].y0//2 + cus[index].height//2,cus[index].x0//2:cus[index].x0//2 + cus[index].width//2]
                                YUVJoint[2][cus[index].y0//2:cus[index].y0//2 + cus[index].height//2,cus[index].x0//2:cus[index].x0//2 + cus[index].width//2] = img2._V[cus[index].y0//2:cus[index].y0//2 + cus[index].height//2,cus[index].x0//2:cus[index].x0//2 + cus[index].width//2]
                        if (position_cu >= nbCUinCTU-1):
                            dec_state = MDC_DEC_STATE.FRAME_WRITE
                            break
                elif (dec_state == MDC_DEC_STATE.FRAME_WRITE):
                    P1 = skimage.metrics.peak_signal_noise_ratio(imgO._Y,img1._Y)
                    P2 = skimage.metrics.peak_signal_noise_ratio(imgO._Y,img2._Y)
                    P0 = skimage.metrics.peak_signal_noise_ratio(imgO._Y,YUVJoint[0])
                    print ("WRITE_FRAME %s PNSR1: %s PSNR2: %s PSNR0: %s" %(frame,P1,P2,P0))
                    
                    rgbD0 = convertYUVToBGR(YUVJoint[0],YUVJoint[2],YUVJoint[1])
                    rgbD1 = convertYUVToBGR(img1._Y,img1._V,img1._U)
                    rgbD2 = convertYUVToBGR(img2._Y,img2._V,img2._U)

                    rgbAll = np.hstack((rgbD0,rgbD1,rgbD2))
                    cv2.imshow("D0 - D1 - D2",rgbAll)
                    cv2.waitKey(0)

                    YUVJoint[0] = YUVJoint[0].ravel()
                    YUVJoint[1] = YUVJoint[1].ravel()
                    YUVJoint[2] = YUVJoint[2].ravel()

                    if (frame == 0):
                        writenpArrayToFile(YUVJoint,yuvD0_file,'wb')
                    else:
                        writenpArrayToFile(YUVJoint,yuvD0_file,'ab')
                    dec_state = MDC_DEC_STATE.SEEK_FRAME
                    frame = frame + 1
                if (frame >= nbFrameToDecode):
                    print ("BREAK")
                    cv2.destroyAllWindows()
                    break
