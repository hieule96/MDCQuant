import numpy as np
import Quadtreelib as qd
import Transform as tf
import random as random
import DPCM
from huffman import huffman_compute

#Default quantification matrix
MQ8x8 = np.zeros((8,8))
MQ8x8 = [[16,16,16,16,17,18,21,24],
[16,16,16,16,17,19,22,25],
[16,16,17,18,20,22,25,29],
[16,16,18,21,24,27,31,36],
[17,17,20,24,30,35,41,47],
[18,19,22,27,35,44,54,65],
[21,22,25,31,41,54,70,88],
[24,25,29,36,47,65,88,115]]

def matq(a,b,Q):
    Quant_matrix = np.zeros((a,b))
    # if (Q<50):
    #     S = 5000/Q
    # else:
    #     S = 200 - 2*Q
    # QuantMatrix_Qfactor = np.floor((S*np.array(MQ8x8)+50)/100)
    # QuantMatrix_Qfactor[QuantMatrix_Qfactor==0] = 1
    for i in range(8):
        for j in range (8):
            for k in range (int (a/8)):
                for l in range (int (b/8)):
                    Quant_matrix[int (a/8)*i+k][int (b/8)*j+l] = ((MQ8x8[i][j])/16)*Q
    return Quant_matrix
def QuantDCT8x8(img, Q, a, b):
    #Matrice dans laquelle on met la future nouvelle image
    newimg = np.zeros((a,b))
    imgdct = np.zeros((a,b))
    #Table de quantification pour un bloc 8x8
    mQ = matq(8,8,Q)
    for i in range(int (a/8)):
        for j in range(int (b/8)):
            imgij = img[i*8:(i+1)*8,j*8:(j+1)*8] #Découpage de l'image originale en blocs de 8x8
            imgijdq = np.around(tf.dct(imgij)/mQ) #Arrondi de la DCT après-quantification
            imgdct[i*8:(i+1)*8,j*8:(j+1)*8] = imgijdq
            newimg[i*8:(i+1)*8,j*8:(j+1)*8] = tf.idct(imgijdq*mQ) #Calcul d'un bloc 8x8 finale (IDCT)
    return newimg, imgdct
def CalRate8x8(imgdct,a,b):
    taille_dct = 0
    imgdct = imgdct.astype(int)
    for k in range(int (a/8)):
        for l in range(int (b/8)):
            (f2, bi2) = huffman_compute(imgdct[k*8:(k+1)*8,l*8:(l+1)*8])
            unique_dct = np.unique(imgdct[k*8:(k+1)*8,l*8:(l+1)*8])
            for j in range(np.size(unique_dct)):
                taille_dct = taille_dct + f2[unique_dct[j]]*np.size(bi2[unique_dct[j]])
    return taille_dct
def QuantDCT8x8Rate(img,a, b,C_Rate,C_Rdelta,MIN_Q=0.001,MAX_Q=100.0):
    #Matrice dans laquelle on met la future nouvelle image
    newimg = np.zeros((a,b))
    imgdct = np.zeros((a,b))
    #Table de quantification pour un bloc 8x8
    Rate = 0.0
    while(Rate>C_Rate+C_Rdelta or Rate<C_Rate-C_Rdelta):
        Q = (MIN_Q+MAX_Q)/2
        mQ = matq(8,8,Q)
        for i in range(int (a/8)):
            for j in range(int (b/8)):
                imgij = img[i*8:(i+1)*8,j*8:(j+1)*8] #Découpage de l'image␣originale en blocs de 8x8
                imgijdq = np.around(tf.dct(imgij)/mQ) #Arrondi de la DCT après␣quantification
                imgdct[i*8:(i+1)*8,j*8:(j+1)*8] = imgijdq
                newimg[i*8:(i+1)*8,j*8:(j+1)*8] = tf.idct(imgijdq*mQ) #Calcul d'un␣bloc 8x8 finale (IDCT)\
        Rate=CalRate8x8(imgdct,a,b)
        if (Rate > C_Rate+C_Rdelta):
            MIN_Q = Q
        else:
            MAX_Q = Q
        print ("Optimizing:"+str(Q)+" %target:"+str(np.around(Rate*100/C_Rate,4)),end='\r', flush=True)
    return newimg, imgdct, Rate
def LCU_QuantDCTD1D2(LCU,QBad,QGood):
    newimgD1 = np.zeros((LCU.h,LCU.w))
    newimgD2 = np.zeros((LCU.h,LCU.w))
    newimgDC = np.zeros((LCU.h,LCU.w))
    imgdctDC = np.zeros((LCU.h,LCU.w))
    imgdctD1 = np.zeros((LCU.h,LCU.w))
    imgdctD2 = np.zeros((LCU.h,LCU.w))
    for i in range (LCU.nbCTU):
        c = qd.find_children(LCU.CTUs[i].root)
        for n in c:
            mQbad = matq(n.get_width(),n.get_height(),QBad)
            mQGood = matq(n.get_width(),n.get_height(),QGood)
            imgdct_cu = tf.dct(n.get_points(LCU.img))
            imgDqGood = np.around(imgdct_cu/mQGood)
            imgDqBad = np.around(imgdct_cu/mQbad)
            cu_x = n.get_x()
            cu_y = n.get_y()
            cu_h = n.get_height()
            cu_w = n.get_width()
            imgdctDC[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w]=imgDqGood
            newimgDC[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDqGood*mQGood)
            i = random.randint(0,100)
            if (n.get_width() > 8 and n.get_height()>8):
                if (i%2 == 0):
                    imgdctD1[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDqBad
                    imgdctD2[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDqGood
                    newimgD1[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDqBad*mQbad)
                    newimgD2[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDqGood*mQGood)
                else:
                    imgdctD1[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDqGood
                    imgdctD2[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDqBad
                    newimgD1[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDqGood*mQGood)
                    newimgD2[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDqBad*mQbad)
            else:
                imgdctD1[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDqGood
                imgdctD2[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDqGood
                newimgD1[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDqGood*mQGood)
                newimgD2[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDqGood*mQGood)
    return imgdctDC,imgdctD1,imgdctD2,newimgDC,newimgD1,newimgD2
# def CalRateQt(imgdct,Quadtree):
#     taille_dct = 0
#     imgdct = imgdct.astype(int)
#     c = qd.find_children(Quadtree.root)
#     #print("Number of segments: %d" %len(c))
#     for n in c:
#         (f2, bi2) = huffman_compute(imgdct[n.get_x():n.get_x()+n.get_width(),n.get_y():n.get_y()+n.get_height()])
#         unique_dct = np.unique(imgdct[n.get_x():n.get_x()+n.get_width(),n.get_y():n.get_y()+n.get_height()])
#         for j in range(np.size(unique_dct)):
#             taille_dct = taille_dct + f2[unique_dct[j]]*np.size(bi2[unique_dct[j]])
#     return taille_dct
def LCU_CalRateQt(LCU,imgdct,AC_Remove = True):
    taille_dct = 0
    imgdct = imgdct.astype(int)
    for ctu in LCU.CTUs:
        for n in ctu:
            cu_x = n.get_x()
            cu_y = n.get_y()
            cu_h = n.get_height()
            cu_w = n.get_width()
            imgdct_CU = imgdct[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w].ravel()
            imgdct_AC = imgdct_CU[1:] if AC_Remove == True else imgdct_CU
            # (f, bi) = huffman_compute(imgdct_AC)
            unique,counts =np.unique(imgdct_AC,return_counts=True)
            probalitity = counts/len(imgdct_AC)
            entropy = -np.sum(probalitity*np.log2(probalitity))
            taille_dct = taille_dct + entropy*(len(imgdct_AC))
            # for j in range(np.size(unique_dct)):
            #     taille_dct = taille_dct + f[unique_dct[j]]*np.size(bi[unique_dct[j]])
    return taille_dct
def LCU_CalRateQt_DC_DPCM(LCU,imgdct):
    imgdct = imgdct.astype(int)
    DC_component_Img = []
    for ctu in LCU.CTUs:
        for cu in ctu:
            cu_x = cu.get_x()
            cu_y = cu.get_y()
            DC_component_Img.append(imgdct[cu_y][cu_x])
    print("Number of element DC %s", len(DC_component_Img))
    half_table = np.array([1,2,4,8,16,32,64,128])
    diff_table = np.hstack([half_table,-half_table])
    DPCM_codec = DPCM.DPCM(diff_table)
    symbol = DPCM_codec.encode(DC_component_Img)
    symbol_diff0 = symbol[symbol!=0]
    symbol_ones = len(symbol) - len(symbol_diff0)
    #Calculate Entropy:
    nb_bit_symbol = np.log2(symbol_diff0)
    return np.sum(nb_bit_symbol) + symbol_ones
    

def LCU_QuantDCTQ1Q2(LCU):
    newimgD1 = np.zeros((LCU.h,LCU.w))
    newimgD2 = np.zeros((LCU.h,LCU.w))
    newimgDC = np.zeros((LCU.h,LCU.w))
    imgdctDC = np.zeros((LCU.h,LCU.w))
    imgdctD1 = np.zeros((LCU.h,LCU.w))
    imgdctD2 = np.zeros((LCU.h,LCU.w))
    i = 0
    for ctu in LCU.CTUs:
        cus = qd.find_children(ctu.root)
        j = 0
        for cu in cus:
            matQ1 = matq(cu.get_width(),cu.get_height(),LCU.Qi1[i][j])
            matQ1[0,0] = 15
            matQ2 = matq(cu.get_width(),cu.get_height(),LCU.Qi2[i][j])
            matQ2[0,0] = 15
            img_dct_cu = tf.dct(cu.get_points(LCU.img))
            # matQ1 = np.full((cu.get_width(),cu.get_height()),LCU.Qi1[i][j])
            # matQ2 = np.full((cu.get_width(),cu.get_height()),LCU.Qi2[i][j])
            # matQ1 = 15
            # matQ2 = 15
            imgDi1 = np.around(img_dct_cu/matQ1)
            imgDi2 = np.around(img_dct_cu/matQ2)
            cu_x = cu.get_x()
            cu_y = cu.get_y()
            cu_h = cu.get_height()
            cu_w = cu.get_width()
            print(i,j,cu_x,cu_y,cu_h,cu_w,len(imgDi1))
            imgdctD1[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDi1
            imgdctD2[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDi2
            imgdctDC[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDi1 if LCU.Qi1[i][j]<LCU.Qi2[i][j] else imgDi2
            
            newimgDC[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDi1*matQ1 if LCU.Qi1[i][j]<LCU.Qi2[i][j] else imgDi2*matQ2)
            newimgD1[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDi1*matQ1)
            newimgD2[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDi2*matQ2)
            j=j+1
        i=i+1
    return imgdctDC,imgdctD1,imgdctD2,newimgDC,newimgD1,newimgD2
def convert_QP_Q(QP):
    Q_value = 2**((1/6)*(QP-4))
    return Q_value
def LCU_QuantDCTQP1QP2(LCU):
    newimgD1 = np.zeros((LCU.h,LCU.w))
    newimgD2 = np.zeros((LCU.h,LCU.w))
    newimgDC = np.zeros((LCU.h,LCU.w))
    imgdctDC = np.zeros((LCU.h,LCU.w))
    imgdctD1 = np.zeros((LCU.h,LCU.w))
    imgdctD2 = np.zeros((LCU.h,LCU.w))
    i = 0
    for ctu in LCU.CTUs:
        j = 0
        for cu in ctu:
            matQ1 = matq(cu.get_width(),cu.get_height(),convert_QP_Q(LCU.Qi1[i][j]))
            matQ1[0,0] = 15
            matQ2 = matq(cu.get_width(),cu.get_height(),convert_QP_Q(LCU.Qi2[i][j]))
            matQ2[0,0] = 15
            img_dct_cu = tf.dct(cu.get_points(LCU.img))
            # matQ1 = np.full((cu.get_width(),cu.get_height()),LCU.Qi1[i][j])
            # matQ2 = np.full((cu.get_width(),cu.get_height()),LCU.Qi2[i][j])
            # matQ1 = 15
            # matQ2 = 15
            imgDi1 = np.around(img_dct_cu/matQ1)
            imgDi2 = np.around(img_dct_cu/matQ2)
            cu_x = cu.get_x()
            cu_y = cu.get_y()
            cu_h = cu.get_height()
            cu_w = cu.get_width()
            #print(i,j,cu_x,cu_y,cu_h,cu_w,len(imgDi1))
            imgdctD1[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDi1
            imgdctD2[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDi2
            imgdctDC[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDi1 if LCU.Qi1[i][j]<LCU.Qi2[i][j] else imgDi2
            
            newimgDC[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDi1*matQ1 if LCU.Qi1[i][j]<LCU.Qi2[i][j] else imgDi2*matQ2)
            newimgD1[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDi1*matQ1)
            newimgD2[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDi2*matQ2)
            j=j+1
        i=i+1
    return imgdctDC,imgdctD1,imgdctD2,newimgDC,newimgD1,newimgD2
