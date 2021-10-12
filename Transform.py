import scipy.fftpack as fft
import numpy as np
def dct(a):
    a = fft.dct(fft.dct(a.T, type=2, norm = 'ortho').T, type=2, norm = 'ortho')
    return a
#Fait la DCT matrice inverse
def idct(a):
    a = fft.idct(fft.idct(a.T, type=2, norm = 'ortho').T, type=2, norm = 'ortho')
    return a

def dst4x4_matrix(a,b,c,d):
    array = [[a,b,c,d],
             [c,c,0,-c],
             [d,-a,-c,b],
             [b,-d,c,-a]]
    return np.array(array,dtype=np.int32)
def dct8x8_matrix(a,b,c,d,e,f,g):
    array = [[ a,  a,  a,  a,  a,  a,  a,  a], 
              [ d,  e,  f,  g, -g, -f, -e, -d], 
              [ b,  c, -c, -b, -b, -c,  c,  b], 
              [ e, -g, -d, -f,  f,  d,  g, -e], 
              [ a, -a, -a,  a,  a, -a, -a,  a], 
              [ f, -d,  g,  e, -e, -g,  d, -f], 
              [ c, -b,  b, -c, -c,  b, -b,  c], 
              [ g, -f,  e, -d,  d, -e,  f, -g]]
    return np.array(array,dtype=np.int32)
def dct16x16_matrix(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o):
    array = [[ a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a],
            [ h,  i,  j,  k,  l,  m,  n,  o, -o, -n, -m, -l, -k, -j, -i, -h],
            [ d,  e,  f,  g, -g, -f, -e, -d, -d, -e, -f, -g,  g,  f,  e,  d],
            [ i,  l,  o, -m, -j, -h, -k, -n,  n,  k,  h,  j,  m, -o, -l, -i], 
            [ b,  c, -c, -b, -b, -c,  c,  b,  b,  c, -c, -b, -b, -c,  c,  b], 
            [ j,  o, -k, -i, -n,  l,  h,  m, -m, -h, -l,  n,  i,  k, -o, -j], 
            [ e, -g, -d, -f,  f,  d,  g, -e, -e,  g,  d,  f, -f, -d, -g,  e], 
            [ k, -m, -i,  o,  h,  n, -j, -l,  l,  j, -n, -h, -o,  i,  m, -k], 
            [ a, -a, -a,  a,  a, -a, -a,  a,  a, -a, -a,  a,  a, -a, -a,  a], 
            [ l, -j, -n,  h, -o, -i,  m,  k, -k, -m,  i,  o, -h,  n,  j, -l], 
            [ f, -d,  g,  e, -e, -g,  d, -f, -f,  d, -g, -e,  e,  g, -d,  f], 
            [ m, -h,  l,  n, -i,  k,  o, -j,  j, -o, -k,  i, -n, -l,  h, -m], 
            [ c, -b,  b, -c, -c,  b, -b,  c,  c, -b,  b, -c, -c,  b, -b,  c], 
            [ n, -k,  h, -j,  m,  o, -l,  i, -i,  l, -o, -m,  j, -h,  k, -n], 
            [ g, -f,  e, -d,  d, -e,  f, -g, -g,  f, -e,  d, -d,  e, -f,  g], 
            [ o, -n,  m, -l,  k, -j,  i, -h,  h, -i,  j, -k,  l, -m,  n, -o]]
    return np.array(array,dtype=np.int32)
def dct32x32_matrix(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,A,B,C,D,E):
   array = [[ a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a,  a], 
              [ p,  q,  r,  s,  t,  u,  v,  w,  x,  y,  z,  A,  B,  C,  D,  E, -E, -D, -C, -B, -A, -z, -y, -x, -w, -v, -u, -t, -s, -r, -q, -p], 
              [ h,  i,  j,  k,  l,  m,  n,  o, -o, -n, -m, -l, -k, -j, -i, -h, -h, -i, -j, -k, -l, -m, -n, -o,  o,  n,  m,  l,  k,  j,  i,  h], 
              [ q,  t,  w,  z,  C, -E, -B, -y, -v, -s, -p, -r, -u, -x, -A, -D,  D,  A,  x,  u,  r,  p,  s,  v,  y,  B,  E, -C, -z, -w, -t, -q], 
              [ d,  e,  f,  g, -g, -f, -e, -d, -d, -e, -f, -g,  g,  f,  e,  d,  d,  e,  f,  g, -g, -f, -e, -d, -d, -e, -f, -g,  g,  f,  e,  d], 
              [ r,  w,  B, -D, -y, -t, -p, -u, -z, -E,  A,  v,  q,  s,  x,  C, -C, -x, -s, -q, -v, -A,  E,  z,  u,  p,  t,  y,  D, -B, -w, -r], 
              [ i,  l,  o, -m, -j, -h, -k, -n,  n,  k,  h,  j,  m, -o, -l, -i, -i, -l, -o,  m,  j,  h,  k,  n, -n, -k, -h, -j, -m,  o,  l,  i], 
              [ s,  z, -D, -w, -p, -v, -C,  A,  t,  r,  y, -E, -x, -q, -u, -B,  B,  u,  q,  x,  E, -y, -r, -t, -A,  C,  v,  p,  w,  D, -z, -s], 
              [ b,  c, -c, -b, -b, -c,  c,  b,  b,  c, -c, -b, -b, -c,  c,  b,  b,  c, -c, -b, -b, -c,  c,  b,  b,  c, -c, -b, -b, -c,  c,  b], 
              [ t,  C, -y, -p, -x,  D,  u,  s,  B, -z, -q, -w,  E,  v,  r,  A, -A, -r, -v, -E,  w,  q,  z, -B, -s, -u, -D,  x,  p,  y, -C, -t], 
              [ j,  o, -k, -i, -n,  l,  h,  m, -m, -h, -l,  n,  i,  k, -o, -j, -j, -o,  k,  i,  n, -l, -h, -m,  m,  h,  l, -n, -i, -k,  o,  j], 
              [ u, -E, -t, -v,  D,  s,  w, -C, -r, -x,  B,  q,  y, -A, -p, -z,  z,  p,  A, -y, -q, -B,  x,  r,  C, -w, -s, -D,  v,  t,  E, -u], 
              [ e, -g, -d, -f,  f,  d,  g, -e, -e,  g,  d,  f, -f, -d, -g,  e,  e, -g, -d, -f,  f,  d,  g, -e, -e,  g,  d,  f, -f, -d, -g,  e], 
              [ v, -B, -p, -C,  u,  w, -A, -q, -D,  t,  x, -z, -r, -E,  s,  y, -y, -s,  E,  r,  z, -x, -t,  D,  q,  A, -w, -u,  C,  p,  B, -v], 
              [ k, -m, -i,  o,  h,  n, -j, -l,  l,  j, -n, -h, -o,  i,  m, -k, -k,  m,  i, -o, -h, -n,  j,  l, -l, -j,  n,  h,  o, -i, -m,  k], 
              [ w, -y, -u,  A,  s, -C, -q,  E,  p,  D, -r, -B,  t,  z, -v, -x,  x,  v, -z, -t,  B,  r, -D, -p, -E,  q,  C, -s, -A,  u,  y, -w], 
              [ a, -a, -a,  a,  a, -a, -a,  a,  a, -a, -a,  a,  a, -a, -a,  a,  a, -a, -a,  a,  a, -a, -a,  a,  a, -a, -a,  a,  a, -a, -a,  a], 
              [ x, -v, -z,  t,  B, -r, -D,  p, -E, -q,  C,  s, -A, -u,  y,  w, -w, -y,  u,  A, -s, -C,  q,  E, -p,  D,  r, -B, -t,  z,  v, -x], 
              [ l, -j, -n,  h, -o, -i,  m,  k, -k, -m,  i,  o, -h,  n,  j, -l, -l,  j,  n, -h,  o,  i, -m, -k,  k,  m, -i, -o,  h, -n, -j,  l], 
              [ y, -s, -E,  r, -z, -x,  t,  D, -q,  A,  w, -u, -C,  p, -B, -v,  v,  B, -p,  C,  u, -w, -A,  q, -D, -t,  x,  z, -r,  E,  s, -y], 
              [ f, -d,  g,  e, -e, -g,  d, -f, -f,  d, -g, -e,  e,  g, -d,  f,  f, -d,  g,  e, -e, -g,  d, -f, -f,  d, -g, -e,  e,  g, -d,  f], 
              [ z, -p,  A,  y, -q,  B,  x, -r,  C,  w, -s,  D,  v, -t,  E,  u, -u, -E,  t, -v, -D,  s, -w, -C,  r, -x, -B,  q, -y, -A,  p, -z], 
              [ m, -h,  l,  n, -i,  k,  o, -j,  j, -o, -k,  i, -n, -l,  h, -m, -m,  h, -l, -n,  i, -k, -o,  j, -j,  o,  k, -i,  n,  l, -h,  m], 
              [ A, -r,  v, -E, -w,  q, -z, -B,  s, -u,  D,  x, -p,  y,  C, -t,  t, -C, -y,  p, -x, -D,  u, -s,  B,  z, -q,  w,  E, -v,  r, -A], 
              [ c, -b,  b, -c, -c,  b, -b,  c,  c, -b,  b, -c, -c,  b, -b,  c,  c, -b,  b, -c, -c,  b, -b,  c,  c, -b,  b, -c, -c,  b, -b,  c], 
              [ B, -u,  q, -x,  E,  y, -r,  t, -A, -C,  v, -p,  w, -D, -z,  s, -s,  z,  D, -w,  p, -v,  C,  A, -t,  r, -y, -E,  x, -q,  u, -B], 
              [ n, -k,  h, -j,  m,  o, -l,  i, -i,  l, -o, -m,  j, -h,  k, -n, -n,  k, -h,  j, -m, -o,  l, -i,  i, -l,  o,  m, -j,  h, -k,  n], 
              [ C, -x,  s, -q,  v, -A, -E,  z, -u,  p, -t,  y, -D, -B,  w, -r,  r, -w,  B,  D, -y,  t, -p,  u, -z,  E,  A, -v,  q, -s,  x, -C], 
              [ g, -f,  e, -d,  d, -e,  f, -g, -g,  f, -e,  d, -d,  e, -f,  g,  g, -f,  e, -d,  d, -e,  f, -g, -g,  f, -e,  d, -d,  e, -f,  g], 
              [ D, -A,  x, -u,  r, -p,  s, -v,  y, -B,  E,  C, -z,  w, -t,  q, -q,  t, -w,  z, -C, -E,  B, -y,  v, -s,  p, -r,  u, -x,  A, -D], 
              [ o, -n,  m, -l,  k, -j,  i, -h,  h, -i,  j, -k,  l, -m,  n, -o, -o,  n, -m,  l, -k,  j, -i,  h, -h,  i, -j,  k, -l,  m, -n,  o], 
              [ E, -D,  C, -B,  A, -z,  y, -x,  w, -v,  u, -t,  s, -r,  q, -p,  p, -q,  r, -s,  t, -u,  v, -w,  x, -y,  z, -A,  B, -C,  D, -E]  
              ]
   return np.array(array,dtype=np.int32)
T_dct32x32 = dct32x32_matrix(64,83,36,89,75,50,18,90,87,80,70,57,43,25,9,90,90,88,85,82,78,73,67,61,54,46,38,31,22,13,4)
T_dct16x16 = dct16x16_matrix(64,83,36,89,75,50,18,90,87,80,70,57,43,25,9)
T_dct8x8 = dct8x8_matrix(64,83,36,89,75,50,18)
T_dst4x4 = dst4x4_matrix(29, 55, 74, 84)
def integerDctTransform32x32(img_cu):
    return T_dct32x32.dot(img_cu).dot(T_dct32x32.T) >> 15
def integerDctTransform16x16(img_cu):
    return T_dct16x16.dot(img_cu).dot(T_dct16x16.T) >> 13
def integerDctTransform8x8(img_cu):
    return T_dct8x8.dot(img_cu).dot(T_dct8x8.T) >> 11
def integerDstTransform4x4(img_cu):
    return (T_dst4x4.dot(img_cu)).dot(T_dst4x4.T) >> 9
def IintegerDctTransform32x32(img_dct):
    return T_dct32x32.T.dot(img_dct).dot(T_dct32x32) >> 19
def IintegerDctTransform16x16(img_dct):
    return T_dct16x16.T.dot(img_dct).dot(T_dct16x16) >> 19
def IintegerDctTransform8x8(img_dct):
    return T_dct8x8.T.dot(img_dct).dot(T_dct8x8) >> 19
def IintegerDstTransform4x4(img_dst):
    return T_dst4x4.T.dot(img_dst).dot(T_dst4x4) >> 19

def transformNxN(img_cu,n):
    img_dct = 0
    if (n==4):
        img_dct = integerDstTransform4x4(img_cu)
    elif (n==8):
        img_dct = integerDctTransform8x8(img_cu)
    elif (n==16):
        img_dct = integerDctTransform16x16(img_cu)
    elif (n==32):
        img_dct = integerDctTransform32x32(img_cu)
    return img_dct
def invTransform(dct_cu,n):
    img_cu = 0
    if (n==4):
        img_cu = IintegerDstTransform4x4(dct_cu)
    elif (n==8):
        img_cu = IintegerDctTransform8x8(dct_cu)
    elif (n==16):
        img_cu = IintegerDctTransform16x16(dct_cu)
    elif (n==32):
        img_cu = IintegerDctTransform32x32(dct_cu)
    return img_cu

