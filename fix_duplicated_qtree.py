# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 00:18:12 2021

@author: LE Trung Hieu
"""
import re
from collections import Counter

old_frame = -1
writeLock = True
pos_end = 79
pos_actual = 0

my_dict = []

with open("decoder_cupu_RondPoint.txt",'r') as fileread:
    with open("decoder_cupu_RondPoint_fixed.txt",'w') as filewrite:
        for lines in fileread:
            ParseTxt = lines
            matchObj  = re.sub('[<>]',"",ParseTxt)      
            matchObj  = re.sub('[ ]',",",matchObj)  
            chunk = matchObj.split(',')
            frame = int(chunk[0])
            pos = int(chunk[1])
            if (pos==0):
                if (frame==old_frame):
                    writeLock=False
                else:
                    writeLock=True
                    old_frame = old_frame + 1
            if (writeLock and frame >= old_frame):
                filewrite.write(lines)
                my_dict.append(frame)
a = dict(Counter(my_dict))

