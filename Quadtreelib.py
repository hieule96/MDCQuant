import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import math
import datetime
import re
#utilities
def printI(img,title):
    fig= plt.figure(figsize=(20, 20))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.title(title)
    plt.show()
def printI2(i1, i2):
    fig= plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(cv2.cvtColor(i1, cv2.COLOR_BGR2RGB))
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(cv2.cvtColor(i2, cv2.COLOR_BGR2RGB))
    plt.show()
class Node():
    def __init__(self, x0, y0, w, h):
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h
        self.children = []
        self.Rcoeff = []
        self.Dcoeff = []
        self.DRcoeff = []
        self.aki = 0
    def set_pos(self,pos):
        self.pos = pos
    def set_Rcoeff(self,Rcoeffs):
        #assert(len(Rcoeffs) == 4,"Number of coefficients must be 4")
        self.Rcoeff = np.array(Rcoeffs)
    def set_Dcoeff(self,Dcoeff):
        #assert(len(Dcoeff) == 4,"Number of coefficients must be 4")
        self.Dcoeff = np.array(Dcoeff)
    def set_DRcoeff(self,DRcoeff):
        #assert(len(DRcoeff) == 3,"Number of coefficients must be 3")
        self.DRcoeff = np.array(DRcoeff)
    def set_aki(self,aki):
        self.aki = aki
    def get_pos(self):
        return self.pos
    def get_width(self):
        return self.width
    def get_height(self):
        return self.height
    def get_points(self,img):
        return img[self.y0:self.y0+self.get_height(),self.x0:self.x0 + self.get_width()]
    def get_x(self):
        return self.x0
    def get_y(self):
        return self.y0
    def rescale_to_real_x_y(self,ctu_x0,ctu_y0):
        self.x0 = self.x0 + ctu_x0
        self.y0 = self.y0 + ctu_y0
    def get_error(self, img):
        pixels = self.get_points(img)
        b_avg = np.mean(pixels[:,:])
        b_mse = np.square(np.subtract(pixels[:,:], b_avg)).mean()
        e =b_mse
        return e * img.shape[0]* img.shape[1]/90000

# class QTree():
#     def __init__(self,x0,y0,w,h,stdThreshold, minPixelSize):
#         self.threshold = stdThreshold
#         self.min_size = minPixelSize
#         self.minPixelSize = minPixelSize
#         self.aki = []
#         self.ctu_x0 = x0
#         self.ctu_y0 = y0
#         self.ctu_w = w
#         self.ctu_h = h
#         self.root = Node(0,0,w,h)
#     def get_points(self,img):
#         return img[self.ctu_y0:self.ctu_y0+self.ctu_h,self.ctu_x0:self.ctu_x0 + self.ctu_w]
#     def subdivide(self,img):
#         recursive_subdivide(self.root, self.threshold, self.minPixelSize,img)
#         c = find_children(self.root)
#         i = 0
#         for n in c:
#             n.set_pos(i)
#             n.rescale_to_real_x_y(self.ctu_x0,self.ctu_y0)
#             self.aki.append((n.get_height()*n.get_width())/(self.ctu_w*self.ctu_h))
#             #print (n.get_height(),n.get_width(),self.ctu_w,self.ctu_h)
#             #print ("aki=",(n.get_height()*n.get_width())/(self.ctu_w*self.ctu_h))
#             i = i+1
#     def render_img(self,img,thickness = 4, color = (255,255,255)):
#         imgc = img.copy()
#         c = find_children(self.root)
#         if thickness > 0:
#             for n in c:
#                 imgc = cv2.rectangle(imgc, (n.x0, n.y0), (n.x0+n.get_width(), n.y0+n.get_height()), color, thickness)
#                 cv2.putText(imgc,str(n.get_pos()),(n.x0, n.x0+n.get_width()),cv2.FONT_HERSHEY_SIMPLEX,1,(36,255,12))
#         return imgc
class LargestCodingUnit:
    def __init__(self,img,threshold,minPixelSize,block_size_w=64,block_size_h=64):
        self.img = img
        self.h = img.shape[0]
        self.w = img.shape[1]
        self.block_size_w = block_size_w
        self.block_size_h = block_size_h
        self.step_h = 0
        self.step_w = 0
        self.CTUs = []
        self.nbCUperCTU = []
        self.tiles = []
        self.nbCTU = 0
        self.threshold = threshold
        self.minPixelSize = minPixelSize
        self.Qi1 = []
        self.Qi2 = []
    def get_img(self,x,y):
        #print ("%s %s %s %s" %(y*self.block_size,y*self.block_size+self.block_size,x*self.block_size,x*self.block_size+self.block_size))
        return self.img[y*self.block_size_h:y*self.block_size_h+self.block_size_h,x*self.block_size_w:x*self.block_size_w+self.block_size_w]
    def repartion_block_64x64(self):
        #Check if image is divisible by
        # assert (self.h%self.block_size_h == 0 and self.w%self.block_size_w == 0)
        self.step_h = int (self.h/self.block_size_h)
        self.step_w = int (self.w/self.block_size_w)
        print ("step_h: " + str(self.step_h))
        print ("step_w: " + str(self.step_w))
        self.nbCTU = self.step_h * self.step_w
        #print ("Aki: node_h, node_w, CTU_h, CTU_w")
        for y in range(0,self.step_h):
            for x in range(0,self.step_w):
                qt = Node(x*self.block_size_w,y*self.block_size_h,self.block_size_w,self.block_size_h)
                recursive_subdivide(qt,self.threshold,self.minPixelSize,self.img)
                self.CTUs.append(qt)
        print("Repartion image in to " + str(self.nbCTU) + " CTU(s)")
    def get_cu(self,index):
        return self.CTUs[index]
    def set_Qi1(self,Qi1_array,i):
        self.Qi1[i] = Qi1_array
    def set_Qi2(self,Qi2_array,i):
        self.Qi2[i] = Qi2_array
    def get_Qi1(self):
        return self.Qi1
    def get_Qi2(self):
        return self.Qi2
    def render_img(self, img,thickness = 1, color = (255,255,255)):
        imgc=img.copy()            
        i = 0
        for ctu in self.CTUs:
            for cu in ctu:
                imgc = cv2.rectangle(imgc, (cu.x0, cu.y0), (cu.x0+cu.get_width(), cu.y0+cu.get_height()), color, thickness)
                # cv2.putText(imgc,str(i),(cu.x0, cu.y0+cu.get_height()),cv2.FONT_HERSHEY_SIMPLEX,.3,color)
                # i=i+1
        return imgc
    def Init_aki(self):
        i = 0
        for ctu in self.CTUs:
            j = 0
            for cu in ctu:
                aki = (cu.get_height()*cu.get_width())/(self.block_size_h*self.block_size_w)
                cu.set_aki(aki)
                j=j+1
                # print (cu.get_height(),cu.get_width(),aki)
            i = i + 1
    def ExportParamtertoCSV(self,name):
        with open('CTU_%s.csv' %(name),'w') as file:
            file.write("x0,y0,x1,y1,Rate Coefficients,Distortion Coefficients,Distortion Rate Coefficients" + '\n')
            for CTU in self.CTUs:
                for node in CTU:
                    file.write("%s,%s,%s,%s,%s,%s,%s" %(node.x0,node.y0,node.x0+node.width,node.y0+node.height,node.Rcoeff,node.Dcoeff,node.DRcoeff))
                    file.write("\n")
        file.close()
    def ExportDRcoefficientToCSV(self,name):
        with open('CTU_%s.csv' %(name),'w') as file:
            for CTU in self.CTUs:
                for node in CTU:
                    file.write("%s" %(node.DRcoeff))
                    file.write("\n")
        file.close()
    def convert_Tree_childrenArray(self):
        for i in range (self.nbCTU):
            self.CTUs[i] = find_children(self.CTUs[i])
    def create_tile(self):
        tile = []
        for ctu_index in range (self.nbCTU):
            for cu in self.CTUs[ctu_index]:
                tile.append(cu)
        self.tiles.append(tile)
    # Merge all cu in a CTU
    def merge_CTU(self):
        CTU = []
        for ctu_index in range (self.nbCTU):
            self.nbCUperCTU.append(len(self.CTUs[ctu_index]))
            for cu in self.CTUs[ctu_index]:
                CTU.append(cu)
        
        self.CTUs = []
        self.nbCTU = 1
        self.CTUs.append(CTU)
        self.block_size_h = self.h
        self.block_size_w = self.w
        self.Init_aki()
    #remove all CU at the bord
    def remove_bord_elements(self,bord_w,bord_h):
        for ctu_index in range (self.nbCTU):
            remove_list = []
            i = 0
            for cu in self.CTUs[ctu_index]:
                if(cu.x0 > bord_w or cu.y0>bord_h or cu.x0+cu.width > bord_w or cu.y0+cu.height > bord_h):
                    remove_list.append(i)
                i = i + 1
            i = 0
            for pos in remove_list:
                # print (ctu_index,remove_list)
                self.CTUs[ctu_index].pop(pos-i)
                i = i + 1
        self.img = self.img[:bord_h,:bord_w]
        self.w = bord_w                
        self.h = bord_h                 
        
def recursive_subdivide(node, k, minPixelSize, img):

    w_1 = int(math.floor(node.width/2))
    w_2 = int(math.ceil(node.width/2))
    h_1 = int(math.floor(node.height/2))
    h_2 = int(math.ceil(node.height/2))
    if node.get_error(img)<=k and w_1<32 and h_1<32:
        return
    if w_1 <= minPixelSize/2 or h_1 <= minPixelSize/2:
        return
    x1 = Node(node.x0, node.y0, w_1, h_1) # top left
    recursive_subdivide(x1, k, minPixelSize, img)
    x2 = Node(node.x0, node.y0+h_1, w_1, h_2) # btm left
    recursive_subdivide(x2, k, minPixelSize, img)
    x3 = Node(node.x0 + w_1, node.y0, w_2, h_1)# top right
    recursive_subdivide(x3, k, minPixelSize, img)
    x4 = Node(node.x0+w_1, node.y0+h_1, w_2, h_2) # btm right
    recursive_subdivide(x4, k, minPixelSize, img)
    node.children = [x1, x2, x3, x4]
def find_children(node):
    if not node.children:
        return [node]
    else:
        children = []
    for child in node.children:
        children += (find_children(child))
    return children
def ComputeQuadTreeYChannel(img_name, threshold=10, minPixelSize=8,x0=0,y0=0,CTU_size_h=64,CTU_size_w=64,img_boarder=20,line_boarder=1, line_color=(255,255,255)):
    imgT= cv2.imread(img_name)
    yvu = cv2.cvtColor(imgT, cv2.COLOR_BGR2YCrCb)
    imgY, v, u = cv2.split(yvu)
    crop_width = int (np.floor(imgY.shape[0]/CTU_size_w)*CTU_size_w)
    crop_heigth =int (np.floor(imgY.shape[1]/CTU_size_h)*CTU_size_h)
    print(crop_width,crop_heigth)
    imgY = imgY[y0:y0+crop_heigth,x0:x0+crop_width]
    printI(imgY,"image")
    lcu = LargestCodingUnit(imgY,threshold,minPixelSize,block_size_h = CTU_size_h,block_size_w=CTU_size_w)
    lcu.repartion_block_64x64()
    lcu.convert_Tree_childrenArray()
    lcu.Init_aki()
    qtImg= lcu.render_img(thickness=line_boarder, color=line_color)
    printI(qtImg,"CTU_size: %sx%s Threshold %s" %(CTU_size_h,CTU_size_w,threshold))
    return lcu,imgY

def import_subdivide(node,subdivide_signal_array,i):
    if (subdivide_signal_array[i]=="99"):
        w_1 = int(math.floor(node.width/2))
        w_2 = int(math.ceil(node.width/2))
        h_1 = int(math.floor(node.height/2))
        h_2 = int(math.ceil(node.height/2))
        x1 = Node(node.x0, node.y0, w_1, h_1) # top left
        x2 = Node(node.x0+w_1, node.y0, w_2, h_1)# top right
        x3 = Node(node.x0, node.y0+h_1, w_1, h_2) # btm 
        x4 = Node(node.x0+w_1, node.y0+h_1, w_2, h_2) # btm right
        i=i+1
        if (subdivide_signal_array[i]=="99"):
            i=import_subdivide(x1, subdivide_signal_array,i)
        i=i+1
        if (subdivide_signal_array[i]=="99"):
            i=import_subdivide(x2, subdivide_signal_array,i)
        i=i+1
        if (subdivide_signal_array[i]=="99"):
            i=import_subdivide(x3, subdivide_signal_array,i)
        i=i+1
        if (subdivide_signal_array[i]=="99"):
            i=import_subdivide(x4, subdivide_signal_array,i)
        node.children = [x1, x2, x3, x4]
    return i


def ImportQuadTreeYchannel(img_name,ctu_file,CTU_perframe = 29):
    imgT= cv2.imread(img_name)
    yvu = cv2.cvtColor(imgT, cv2.COLOR_BGR2YCrCb)
    imgY, v, u = cv2.split(yvu)
    lcu = LargestCodingUnit(imgY.astype(np.float32) - 128,1,8)
    step_w = np.ceil (lcu.w / lcu.block_size_w)
    with open(ctu_file,'r') as file:
        for lines in file:
            ParseTxt = lines
            matchObj  = re.sub('[<>]',"",ParseTxt)      
            matchObj  = re.sub('[ ]',",",matchObj)      
            chunk = matchObj.split(',')
            pos = int(chunk[1])
            quadtree_composition = chunk[2:]
            CTU = Node(int (pos%step_w)*64,int (pos/step_w)*64,64,64)
            import_subdivide(CTU,quadtree_composition,0)
            lcu.CTUs.append(CTU)
            lcu.nbCTU = lcu.nbCTU + 1
            if (pos==CTU_perframe):
                break
    return lcu,imgY
def concat_images(img1, img2, boarder=5, color=(255,255,255)):
    img1_boarder = cv2.copyMakeBorder(
    img1,
    boarder, #top
    boarder, #btn
    boarder, #left
    boarder, #right
    cv2.BORDER_CONSTANT,
    value=color
    )
    img2_boarder = cv2.copyMakeBorder(
    img2,
    boarder, #top
    boarder, #btn
    0, #left
    boarder, #right
    cv2.BORDER_CONSTANT,
    value=color
    )
    return np.concatenate((img1_boarder, img2_boarder), axis=1)
