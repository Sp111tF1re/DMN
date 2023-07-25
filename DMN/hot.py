import cv2
import numpy as np
from models.kspace_loss import DFT
path_lr = "D:/python_code/MRI_SR_ver1_1/models/2_.jpg"
path_y = "D:/python_code/MRI_SR_ver1_1/imgs/0047_lr.png"

img_lr = cv2.imread(path_lr, 0)
img_y = cv2.imread(path_y, 0)
img = img_y-img_lr
# img_lr = np.array(img_lr)
# img_y = np.array(img_y)
# img = np.array(img_lr-img_y)
hot = cv2.absdiff(img_y, img_lr)
# hot.astype(int)
# print(img.min(), img.max())
# print(hot)
hot_img = cv2.applyColorMap(hot, 2)
array_hot = np.array(hot)

cv2.imwrite("1.jpg", hot_img)
# print(hot)

