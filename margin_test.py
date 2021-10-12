#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:57:21 2021

@author: ubuntu
"""
count = 0
with open("QP1.csv","r") as q1File:
        with open("QP1_test_margin.csv","w") as q1FileCopy:
            with open("QP2_test_margin.csv","w") as q2FileCopy:
                for line in q1File:
                    element = line.split(',')
                    for i in range (0,len(element)-1):
                        if ((count%2)==0):
                            q1FileCopy.write("25,")
                            q2FileCopy.write("51,")
                        else:
                            q2FileCopy.write("25,")
                            q1FileCopy.write("51,")
                        count = count + 1
                    q1FileCopy.write("\n")
                    q2FileCopy.write("\n")
