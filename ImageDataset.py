import os
import functools
import torch
import pandas as pd
from PIL import Image
from PIL import ImageFile

import cv2
import time
from torch.utils.data import Dataset
from torchvision import transforms

class select_data(Dataset):
    def __init__(self,img_dir):
        self.img_dir = img_dir
        self.img_dirs = os.listdir(img_dir)

    def __getitem__(self, index):
        img_1 = cv2.imread(os.path.join(self.img_dir,self.img_dirs[index],'0.png'))
        img_2 = cv2.imread(os.path.join(self.img_dir,self.img_dirs[index],'1.png'))
    
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
        trans = transforms.ToTensor()
    
        img_1 = trans(img_1)
        img_2 = trans(img_2)
    
        return img_1, img_2
    def __len__(self):
        return len(self.img_dirs)