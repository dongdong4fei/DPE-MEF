import torch
from test_thorough import Tester
import argparse
import ast
import os
import numpy as np
import random

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', default='./checkpoints/', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default='DEM.pt', type=str, help='name of the checkpoint to load')
    parser.add_argument('--ckpt_path_color', default='./checkpoints/', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt_color', default='CEM.pt', type=str, help='name of the checkpoint to load')
    parser.add_argument('--test_img_path', default='./test_imgs/', type=str,
                        metavar='PATH')
    parser.add_argument('--result_path', default='./results/', type=str,
                        metavar='PATH')


    return parser.parse_args()

def main(cfg):
    t = Tester(cfg)
    t.eval(0)


if __name__ == '__main__':
    config = parse_config()
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main(config)