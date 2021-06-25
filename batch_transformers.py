import numpy as np
from torchvision import transforms
import torch
from PIL import Image
import collections
import random
RANDOM_RESOLUTIONS = [512, 768, 1024, 1280, 1536]



def rand_crop(data,size):
    width1 = random.randint(0, data[0].size[0] - size)
    height1 = random.randint(0, data[0].size[1] - size)
    width2 = width1 + size
    height2 = height1 + size
    i=0
    while i < len(data):
        data[i] = data[i].crop((width1, height1, width2, height2))
        i += 1
    #label = label.crop((width1, height1, width2, height2))
    return data
class BatchRandomResolution(object):
    def __init__(self, size=256, interpolation=Image.BILINEAR):

        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2) or (size is None)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        #imgs = [img[0],img[1]]
        # if self.size is None:
        #     h, w = imgs[0].size
        #     max_idx = 0
        #     for i in range(len(RANDOM_RESOLUTIONS)):
        #         if h > RANDOM_RESOLUTIONS[i] and w > RANDOM_RESOLUTIONS[i]:
        #             max_idx += 1
        #     idx = np.random.randint(max_idx)
        #     self.size = RANDOM_RESOLUTIONS[idx]
        # imgs.append(gt)
        # imgs2 = rand_crop(imgs, self.size)


        # return imgs2
        return [transforms.Resize([self.size, self.size])(img) for img in imgs]

class BatchRandomResolution_test(object):
    def __init__(self, size=None, interpolation=Image.BILINEAR):

        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2) or (size is None)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):

        # if self.size is None:
        #     h, w = imgs[0].size
        #     max_idx = 0
        #     for i in range(len(RANDOM_RESOLUTIONS)):
        #         if h > RANDOM_RESOLUTIONS[i] and w > RANDOM_RESOLUTIONS[i]:
        #             max_idx += 1
        #     idx = np.random.randint(max_idx)
        #     self.size = RANDOM_RESOLUTIONS[idx]
        # #return [transforms.RandomCrop([self.size,self.size])(img) for img in imgs]
        return [transforms.Resize(self.size)(img) for img in imgs]

class BatchTestResolution(object):
    def __init__(self, size=None, interpolation=Image.BILINEAR):

        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2) or (size is None)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):

        # h, w = imgs[0].size
        # if h > self.size and w > self.size:
        #     #return [transforms.RandomCrop(self.size, self.size)(img) for img in imgs]
        #     return [transforms.Resize([size,size])(img) for img in imgs]
        # else:
        #     return imgs
        return [transforms.Resize([144,144])(img) for img in imgs]

class BatchTestResolution(object):
    def __init__(self, size=None, interpolation=Image.BILINEAR):

        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2) or (size is None)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):

        # h, w = imgs[0].size
        # if h > self.size and w > self.size:
        #     #return [transforms.RandomCrop(self.size, self.size)(img) for img in imgs]
        #     return [transforms.Resize([size,size])(img) for img in imgs]
        # else:
        #     return imgs
        return [transforms.Resize([self.size,self.size])(img) for img in imgs]
class BatchToTensor(object):
    def __call__(self, imgs):

        return [transforms.ToTensor()(img) for img in imgs]


class BatchRGBToGray(object):
    def __call__(self, imgs):
        return [img[0, :, :] * 0.299 + img[1, :, :] * 0.587 + img[2:, :, :] * 0.114 for img in imgs]

class RGBToGray(object):
    def __call__(self, img):
        x = img[:,0, :, :] * 0.299 + img[:,1, :, :] * 0.587 + img[:,2, :, :] * 0.114
        return torch.stack((x,x,x),dim=1)


class BatchRGBToYCbCr(object):
    def __call__(self, imgs):
        return [torch.stack((0. / 256. + img[0, :, :] * 0.299000 + img[1, :, :] * 0.587000 + img[2, :, :] * 0.114000,
                           128. / 256. - img[0, :, :] * 0.168736 - img[1, :, :] * 0.331264 + img[2, :, :] * 0.500000,
                           128. / 256. + img[0, :, :] * 0.500000 - img[1, :, :] * 0.418688 - img[2, :, :] * 0.081312),
                          dim=0) for img in imgs]


class YCbCrToRGB(object):
    def __call__(self, img):
        return torch.stack((img[:, 0, :, :] + (img[:, 2, :, :] - 128 / 256.) * 1.402,
                            img[:, 0, :, :] - (img[:, 1, :, :] - 128 / 256.) * 0.344136 - (img[:, 2, :, :] - 128 / 256.) * 0.714136,
                            img[:, 0, :, :] + (img[:, 1, :, :] - 128 / 256.) * 1.772),
                            dim=1)

class RGBToYCbCr(object):
    def __call__(self, img):
        return torch.stack((0. / 256. + img[:, 0, :, :] * 0.299000 + img[:, 1, :, :] * 0.587000 + img[:, 2, :, :] * 0.114000,
                           128. / 256. - img[:, 0, :, :] * 0.168736 - img[:, 1, :, :] * 0.331264 + img[:, 2, :, :] * 0.500000,
                           128. / 256. + img[:, 0, :, :] * 0.500000 - img[:, 1, :, :] * 0.418688 - img[:, 2, :, :] * 0.081312),
                          dim=1)
