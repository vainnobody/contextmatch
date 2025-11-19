import os
import math
import random
import numpy as np
from PIL import Image
from copy import deepcopy

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from dataset.transform import *


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None,  ignore_value=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.ignore_value = ignore_value
        self.reduce_zero_label = True if name == 'ade20k' else False

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        # id = self.ids[item]
        # print("len(self.ids) - 1", len(self.ids) - 1)
        # print("self.ids", self.ids)
        # id = self.ids[len(self.ids) - 1]
        id = random.choice(self.ids)
        # print("self.mode {}, id {}".format(self.mode, id))
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask =Image.open(os.path.join(self.root, id.split(' ')[1]))

        if self.reduce_zero_label:
            mask = np.array(mask)
            mask[mask == 0] = 255
            mask = mask - 1
            mask[mask == 254] = 255
            mask = Image.fromarray(mask)

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask

        ignore_value = 254 if self.mode == 'train_u' else self.ignore_value
        # img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = resize(img, mask, (0.5, 2.0))
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = self.ignore_value

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2, mask

    def __len__(self):
        
        return int(len(self.ids) * 100)
        # return 1000
        # return len(self.ids)
