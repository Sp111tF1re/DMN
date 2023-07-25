import cv2
import numpy as np

path_lr = "D:/python_code/MRI_SR_ver1_1/imgs/0011_lr.png"

def noiseGauss(img, sigma):
	temp_img = np.float64(np.copy(img))
	h = temp_img.shape[0]
	w = temp_img.shape[1]
	noise = np.random.randn(h,w) * sigma
	noisy_img = np.zeros(temp_img.shape, np.float64)
	if len(temp_img.shape) == 2:
		noisy_img = temp_img + noise
	else:
		noisy_img[:,:,0] = temp_img[:,:,0] + noise
	return noisy_img

img = cv2.imread(path_lr, 0)
result = noiseGauss(img, 0)
# result = result.astype(np.uint8)
cv2.imwrite('result.jpg', result)