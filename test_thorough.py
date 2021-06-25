import os
import torch
from torch.utils.data import DataLoader
from ImageDataset import *
import numpy as np
import cv2
from models import DEM,CEM
from batch_transformers import RGBToYCbCr, RGBToGray ,YCbCrToRGB


def save_images(filepath, result_1, result_2=None, result_3=None, result_4=None):
    result_1 = result_1.permute([0, 2, 3, 1]).cpu().detach().numpy()
    result_1 = np.squeeze(result_1)
    result_1 = cv2.cvtColor(result_1, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, result_1 * 255.0)
    print(filepath)


class Tester(object):
    def __init__(self, config):
        self.test_batch_size = 1
        self.epoch = 0
        self.dem = DEM()
        self.cem = CEM()
        self.dem.cuda()
        self.cem.cuda()
        self.RGBToYCbCr = RGBToYCbCr()
        self.RGBToGray = RGBToGray()
        self.YCbCrToRGB = YCbCrToRGB()
        self.config = config

        self.testDataset = select_data(self.config.test_img_path)
        self.test_dataloader = DataLoader(dataset=self.testDataset,
                                           batch_size=self.test_batch_size,
                                           num_workers=6,
                                           shuffle=False)

        if config.ckpt:
            ckpt = os.path.join(config.ckpt_path, config.ckpt)
            ckpt_color = os.path.join(config.ckpt_path_color, config.ckpt_color)
        self._load_checkpoint(ckpt=ckpt)
        self._load_checkpoint2(ckpt=ckpt_color)

    def eval(self, epoch):
        print('-----------------------start eval-------------------------')
        self.dem.eval()
        self.cem.eval()
        path = self.config.test_img_path
        img_dirs = os.listdir(path)
        with torch.no_grad():
            for step, sample_batched in enumerate(self.test_dataloader, 0):

                I_hr = sample_batched
                for i in range(len(I_hr)):
                    I_hr[i] = I_hr[i].cuda()

                img1_ycbcr = self.RGBToYCbCr(I_hr[0])
                img2_ycbcr = self.RGBToYCbCr(I_hr[1])
                #a = time.time()
                O_hr_y = self.dem(torch.cat((I_hr[0], I_hr[1]), dim=1))  # 1 chan or 3 channs
                O_hr_y =  (O_hr_y - torch.min(O_hr_y)) / (torch.max(O_hr_y) - torch.min(O_hr_y))
                generated_cbcr = self.cem(torch.cat((img1_ycbcr, img2_ycbcr, O_hr_y), dim=1))
                #b = time.time()
                #print("time:" + str(b - a))
                O_hr_enhanced = self.YCbCrToRGB(torch.cat((O_hr_y, generated_cbcr), dim=1))
                save_images(os.path.join(self.config.result_path,str(step+1)+'.png'),
                                O_hr_enhanced)

                print('done:' + str(step+1) )
        return





    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.dem.load_state_dict(checkpoint['state_dict'])
            # self.epoch = checkpoint['epoch']
            print("[*] loaded checkpoint '{}' "
                  .format(ckpt))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))
            
    def _load_checkpoint2(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.cem.load_state_dict(checkpoint['state_dict'])
            # self.optimizer_G.load_state_dict(checkpoint['optimizer'])
            self.epoch = checkpoint['epoch']
            # if self.initial_lr is not None:
            # 	for param_group in self.optimizer.param_groups:
            # 		param_group['initial_lr'] = self.initial_lr
            print("[*] loaded checkpoint '{}'"
                  .format(ckpt))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))
