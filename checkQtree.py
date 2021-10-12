# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 12:34:17 2021

@author: LE Trung Hieu
"""

with open("decoder_cupu.txt") as orginal_qtree:
    with open("Qt_QP0.csv.txt") as reconstructed_qtree:
        line_o = orginal_qtree.readline()
        line_re = reconstructed_qtree.readline()
        line = 1
        if (len(line_o) != len(line_re)):
            print ("line %d"%(line))
            print (len(line_o) == len(line_re))
        while (len(line_o)!=0):
            line_o = orginal_qtree.readline()
            line_re = reconstructed_qtree.readline()
            line = line + 1
            if (len(line_o) != len(line_re)):
                print ("line %d"%(line))
                print (len(line_o) == len(line_re))