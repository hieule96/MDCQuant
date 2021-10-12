#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 18:25:35 2021

@author: ubuntu
"""
import numpy as np
from matplotlib import pyplot as plt

class FrameYUV():
    def __init__(self, Y, U, V):
        self._Y = Y
        self._U = U
        self._V = V
    def copy(self):
        return self._Y.copy(),self._U.copy(),self._V.copy()
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

def displayFrameY(frame):
    plt.imshow(frame._Y,cmap='gray')

def writenpArrayToFile(frame,outputFileName,mode='wb'):
    with open(outputFileName,mode) as output:
        output.write(frame._Y.tobytes())
        output.write(frame._U.tobytes())
        output.write(frame._V.tobytes())    
    