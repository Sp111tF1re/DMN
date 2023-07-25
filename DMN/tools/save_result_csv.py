import random
import pandas as pd
from datetime import datetime


def write(psnr, ssim, step):
    if step == 1:
    	df = pd.DataFrame(columns=['time', 'step', 'psnr', 'ssim'])
    	df.to_csv("./results/MRI/metric.csv", mode='a', index=False)
    else:
    	time = "%s" % datetime.now()
    	step = "Step[%d]"%step
    	psnr = "%f"%psnr
    	ssim = "%f"%ssim
    	list = [time, step, psnr, ssim]
    	data = pd.DataFrame([list])
    	data.to_csv("./results/MRI/metric.csv", mode='a', header=False, index=False)
