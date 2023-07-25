import numpy as np
import cv2
from matplotlib import pyplot as plt

path_lr = "D:/python_code/MRI_SR_ver1_1/imgs/0047_lr.png"
path_y = "D:/python_code/MRI_SR_ver1_1/imgs/0047_y.png"

img_lr = cv2.imread(path_lr, 0)
img_y = cv2.imread(path_y, 0)

def DFT(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    result = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return result, dft_shift

def IDFT(dft_shift):
    ishift = np.fft.ifftshift(dft_shift)
    iimg = cv2.idft(ishift)
    iimg = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
    return iimg

def Normalization(img):
    cmax = img.max()
    cmin = img.min()
    img = (img-cmin)/(cmax-cmin)*255
    return img

if __name__ == "__main__":

    dft_lr, shift_lr = DFT(img_lr)
    dft_y, shift_y = DFT(img_y)
    idft_lr = IDFT(shift_lr)
    # print(idft_lr)
    idft_lr = Normalization(idft_lr)
    # print(idft_lr)
    idft_y = IDFT(shift_y)
    idft_y = Normalization(idft_y)
    cv2.imwrite('1.jpg', dft_lr)
    cv2.imwrite('2.jpg', dft_y)
    cv2.imwrite('1_.jpg', idft_lr)
    cv2.imwrite('2_.jpg', idft_y)