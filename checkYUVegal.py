# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:55:49 2021

@author: LE Trung Hieu
"""
import numpy as np
from matplotlib import pyplot as plt
 
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

class FrameYUV(object):
    def __init__(self, Y, U, V):
        self._Y = Y
        self._U = U
        self._V = V

imgY1 = read_YUV420_frame(open("resi.yuv","rb"),352,288,0)._Y
imgY2 = read_YUV420_frame(open("resi2.yuv","rb"),352,288,0)._Y
plt.imshow(imgY2,cmap='gray')