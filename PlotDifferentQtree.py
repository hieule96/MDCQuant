#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 12:53:50 2021

@author: ubuntu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Statupdatedr0-8-10.csv").round(3)
df2 = pd.read_csv("Stat_Encoder.csv").round(3)
plt.scatter(df2['Bits(Mbits)'],df2['MSEY'])
plt.show()
plt.scatter(df2['QP'],df2['MSEY'])
plt.show()
plt.scatter(df2['QP'],df2['Bits(Mbits)']*1e6/(352*288))
plt.show()
bestPSNR0_list_bylambda = []
bestqtreebyQP_list_bylambda = []
bestRatebylambda = []
lambda_list = []
for d in df.groupby(df['lambda']):
    lambda_list.append(d[0])
    line1 = plt.scatter(d[1]['R0(Mbits)'],d[1]['PSNR0'],label="MDC")
    line2 = plt.scatter(df2['Bits(Mbits)']*2,df2['PSNRY'],label="HEVC")
    bestperformanceidx = d[1]['PSNR0'].idxmax()
    bestR = d[1]['R0(Mbits)'][bestperformanceidx]
    bestPSNR0 = d[1]['PSNR0'][bestperformanceidx]
    bestPSNR0_list_bylambda.append(bestPSNR0)
    bestQtreeQP = d[1]['QP'][bestperformanceidx]
    bestqtreebyQP_list_bylambda.append(bestQtreeQP)
    bestRatebylambda.append(bestR)
    maxPSNR= df2['PSNRY'].max() if df2['PSNRY'].max() > bestPSNR0 else bestPSNR0
    maxBit = df2['Bits(Mbits)'].max() if df2['Bits(Mbits)'].max()>bestR else bestR
    df2['diffwithbest'] = ((df2['Bits(Mbits)']*2 - bestR)/maxBit)**2
    deltaminidx = df2['diffwithbest'].idxmin()
    deltamin = df2['PSNRY'][deltaminidx] - bestPSNR0
    pointCompare = plt.plot(df2['Bits(Mbits)'][deltaminidx]*2,df2['PSNRY'][deltaminidx],label="Ref R:%s PSNR:%s QP:%d" %(df2['Bits(Mbits)'][deltaminidx]*2,df2['PSNRY'][deltaminidx],deltaminidx),marker="X",color='r')
    pointMDC = plt.plot(bestR,bestPSNR0,marker="X",color='r',label=r'R:%s PSNR:%s QP:%d $\Delta$min:%s' %(bestR,bestPSNR0,bestQtreeQP,np.round(deltamin,3)) )
    plt.legend()
    plt.yscale("log")  
    plt.title("lamda MDC %s" %d[0])
    plt.xlabel("Rate in Mbits")
    plt.ylabel("PNSR (dB)")
    plt.show()
for d in df.groupby(df['QP']):
    line3 = plt.plot(df2['Bits(Mbits)'],df2['PSNRY'],label="HEVC/R",)
    line1 = plt.scatter(d[1]['R0(Mbits)'],d[1]['PSNR0'],label="MDC")
    line2 = plt.plot(df2['Bits(Mbits)']*2,df2['PSNRY'],label="HEVC/2R")
    plt.legend()
    plt.title("QP %s" %d[0])
    plt.xlabel("Rate in Mbits")
    plt.ylabel("PNSR (dB)")
    plt.yscale("log")  
    plt.show()
df = pd.read_csv("Statupdatedr1-8-10.csv").round(3)
df2 = pd.read_csv("Stat_Encoder.csv").round(3)
plt.scatter(df2['Bits(Mbits)'],df2['MSEY'])
plt.show()
plt.scatter(df2['QP'],df2['MSEY'])
plt.show()
plt.scatter(df2['QP'],df2['Bits(Mbits)']*1e6/(352*288))
plt.show()
bestPSNR0r1_list_bylambda = []
bestqtreebyQP_list_bylambda_r1 = []
bestRatebylambdar1 = []
lambdar1_list = []
for d in df.groupby(df['lambda']):
    lambdar1_list.append(d[0])
    line1 = plt.scatter(d[1]['R0(Mbits)'],d[1]['PSNR0'],label="MDC")
    line2 = plt.scatter(df2['Bits(Mbits)']*2,df2['PSNRY'],label="HEVC")
    bestperformanceidx = d[1]['PSNR0'].idxmax()
    bestR = d[1]['R0(Mbits)'][bestperformanceidx]
    bestPSNR0 = d[1]['PSNR0'][bestperformanceidx]
    bestPSNR0r1_list_bylambda.append(bestPSNR0)
    bestQtreeQP = d[1]['QP'][bestperformanceidx]
    bestqtreebyQP_list_bylambda_r1.append(bestQtreeQP)
    bestRatebylambdar1.append(bestR)
    maxPSNR= df2['PSNRY'].max() if df2['PSNRY'].max() > bestPSNR0 else bestPSNR0
    maxBit = df2['Bits(Mbits)'].max() if df2['Bits(Mbits)'].max()>bestR else bestR
    df2['diffwithbest'] = ((df2['Bits(Mbits)']*2 - bestR)/maxBit)**2
    deltaminidx = df2['diffwithbest'].idxmin()
    deltamin = df2['PSNRY'][deltaminidx] - bestPSNR0
    pointCompare = plt.plot(df2['Bits(Mbits)'][deltaminidx]*2,df2['PSNRY'][deltaminidx],label="Ref R:%s PSNR:%s QP:%d" %(df2['Bits(Mbits)'][deltaminidx]*2,df2['PSNRY'][deltaminidx],deltaminidx),marker="X",color='r')
    pointMDC = plt.plot(bestR,bestPSNR0,marker="X",color='r',label=r'R:%s PSNR:%s QP:%d $\Delta$min:%s' %(bestR,bestPSNR0,bestQtreeQP,np.round(deltamin,3)) )
    plt.legend()
    plt.yscale("log")  
    plt.title("lamda MDC %s" %d[0])
    plt.xlabel("Rate in Mbits")
    plt.ylabel("PNSR (dB)")
    plt.show()
    
for d in df.groupby(df['QP']):
    line3 = plt.plot(df2['Bits(Mbits)'],df2['PSNRY'],label="HEVC/R",)
    line1 = plt.scatter(d[1]['R0(Mbits)'],d[1]['PSNR0'],label="MDC")
    line2 = plt.plot(df2['Bits(Mbits)']*2,df2['PSNRY'],label="HEVC/2R")
    plt.legend()
    plt.title("QP %s" %d[0])
    plt.xlabel("Rate in Mbits")
    plt.ylabel("PNSR (dB)")
    plt.yscale("log")  
    plt.show()
    
plt.title("Comparison of MDC with differents redundancy")
plt.scatter(bestRatebylambdar1,bestPSNR0r1_list_bylambda,color='c',label="Best MDC central PSNR (r=1.0)")
for i in range (0,len(bestRatebylambdar1)):
    plt.annotate(lambdar1_list[i], (bestRatebylambdar1[i],bestPSNR0r1_list_bylambda[i]),color="c")
plt.scatter (bestRatebylambda,bestPSNR0_list_bylambda,color='r',label="Best MDC central PSNR (r=0.0)")
plt.scatter (df2['Bits(Mbits)']*2,df2['PSNRY'],color='b',label="HEVC PSNR/2R")
plt.scatter (df2['Bits(Mbits)'],df2['PSNRY'],color='g',label="HEVC PSNR/R")
for i in range (0,len(bestRatebylambda)):
    plt.annotate(lambda_list[i], (bestRatebylambda[i],bestPSNR0_list_bylambda[i]),color="r")
plt.xlabel("Mbits")
plt.ylabel("PNSR (dB)")
plt.legend()
plt.show()

