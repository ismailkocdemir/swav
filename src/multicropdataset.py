# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger

from PIL import ImageFilter
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import kornia.augmentation as K
from kornia.filters import GaussianBlur2d as KorniaGaussianBlur

logger = getLogger()


class MultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
    ):
        super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class KorniaRandomGaussianBlur(nn.Module):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, kernel_size, sigma, p):
        super(KorniaRandomGaussianBlur, self).__init__()
        self.prob = p
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.blur = KorniaGaussianBlur(kernel_size, sigma)

    def forward(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img
        return self.blur(img)


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

class KorniaAugmentationPipeline(nn.Module):
   def __init__(self,
                s_color=1.0, 
                p_color=0.8, 
                p_flip=0.5,
                p_gray=0.2, 
                p_blur=0.5, 
                kernel_min=0.1, 
                kernel_max=2.) -> None:
      super(KorniaAugmentationPipeline, self).__init__()
      
      #T_hflip = K.RandomHorizontalFlip(p=p_flip) 
      #T_gray = K.RandomGrayscale(p=p_gray)
      self.T_color = K.ColorJitter(p_color, 0.8*s_color, 0.8*s_color, 0.8*s_color, 0.2*s_color)

      #radius = kernel_max*2  # https://dsp.stackexchange.com/questions/10057/gaussian-blur-standard-deviation-radius-and-kernel-size
      #kernel_size = int(radius*2 + 1) # needs to be odd.
      #kernel_size = (kernel_size, kernel_size)

      #self.T_blur = KorniaRandomGaussianBlur(kernel_size=kernel_size, sigma=(kernel_min, kernel_max), p=p_blur)
      '''
      self.transform = nn.Sequential(
            T_hflip,
            T_color,
            T_gray,
      )
      '''

   def forward(self, input):
       #out = self.transform(input)
       #return self.T_blur(out[0])
       return self.T_color(input)[0]
      