import os
import random
from datetime import datetime

import numpy as np
import argparse
from yacs.config import CfgNode
import torch
import torch.distributed as dist

from tools.pseudo_face_data import faces_data
from tools.utils import save_tensor_image, AverageMeter
from models.face_model import Face_Model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
main_parse = argparse.ArgumentParser()
main_parse.add_argument("yaml", type=str)
main_parse.add_argument("--port", type=int, default=2357, required=False)
main_args = main_parse.parse_args()

with open(main_args.yaml, "rb") as cf:
    CFG = CfgNode.load_cfg(cf)
    CFG.freeze()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_args.port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, cpu=False):
    if cpu:
        rank = torch.device("cpu")
    elif world_size > 1:
        setup(rank, world_size)
    model = Face_Model(rank, CFG, world_size > 1)
    model.net_load("./results/MRI/nets/nets_28000.pth")

    testset = faces_data(data_lr=os.path.join(CFG.DATA.FOLDER, "TEST/LR"), data_hr = None, b_train=False, shuffle=False, img_range=CFG.DATA.IMG_RANGE, rgb=CFG.DATA.RGB)
    img_save_folder = os.path.join(CFG.EXP.OUT_DIR, "imgs")
    os.makedirs(img_save_folder, exist_ok=True)
    print("Output dir: ", CFG.EXP.OUT_DIR)
    print(f"\nTesting and saving: {datetime.now().strftime('%Y-%d-%m_%H:%M:%S')}")
    model.mode_selector("eval")
    for b in range(len(testset)):
        if b > 10000:
            break
        else:
            lr = testset[b]["lr"].unsqueeze(0).to(rank)
            y, _ = model.test_sample(lr)
            save_tensor_image(os.path.join(img_save_folder, f"{b:04d}_y.png"), y, CFG.DATA.IMG_RANGE, CFG.DATA.RGB)
            save_tensor_image(os.path.join(img_save_folder, f"{b:04d}_lr.png"), lr, CFG.DATA.IMG_RANGE, CFG.DATA.RGB)


if __name__ == "__main__":
    random_seed = CFG.EXP.SEED
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    n_gpus = 0
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()

    main(0, n_gpus, cpu=(n_gpus == 0))
    print("fin.")

