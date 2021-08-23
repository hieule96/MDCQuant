import scipy.fftpack as fft

def dct(a):
    a = fft.dct(fft.dct(a.T, type=2, norm = 'ortho').T, type=2, norm = 'ortho')
    return a
#Fait la DCT matrice inverse
def idct(a):
    a = fft.idct(fft.idct(a.T, type=2, norm = 'ortho').T, type=2, norm = 'ortho')
    return a