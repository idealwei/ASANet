import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import cv2


class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, img_size=(1024, 512), norm=False, random_mirror=False, random_crop=False, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.img_size = img_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.random_mirror = random_mirror
        self.random_crop = random_crop
        if not max_iters == None:
            self.img_ids = self.img_ids * \
                int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.set = set
        self.norm = norm
        self.void_classes = [0, 1, 2, 3, 4, 5,
                             6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19,
                              20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_map = dict(zip(self.valid_classes, range(19)))
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(
                self.root, "leftImg8bit/%s/%s" % (self.set, name))
            lbl_file = osp.join(self.root, "gtFine/%s/%s" % (self.set,
                                                             name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")))
            self.files.append({
                "img": img_file,
                "lbl": lbl_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["lbl"])
        # resize

        if self.random_crop:
            img_w, img_h = image.size
            crop_w, crop_h = self.img_size
            h_off = random.randint(0, img_h - crop_h)
            w_off = random.randint(0, img_w - crop_w)
            image = image.crop((w_off, h_off, w_off+crop_w, h_off+crop_h))
            label = label.crop((w_off, h_off, w_off+crop_w, h_off+crop_h))
        else:
            image = image.resize(self.img_size, Image.BICUBIC)
            label = label.resize(self.img_size, Image.NEAREST)
        image = np.asarray(image, np.float32)
        label = self.encode_segmap(np.array(label, dtype=np.uint8))
        size = image.shape[:2]
        # normalize image
        if self.random_mirror:
            flip = np.random.choice(2)*2 - 1
            image = image[:, ::flip, :]
            label = label[:, ::flip]

        if self.norm:
            image = image / 255.0
            image -= np.array([0.485, 0.456, 0.406])
            image = image / np.array([0.229, 0.224, 0.225])
        else:
            image = image - np.array([122.67892, 116.66877, 104.00699])
        image = image.transpose((2, 0, 1)).astype(np.float32)
        return image.copy(), label.copy(), np.array(size), name

    def encode_segmap(self, mask):
        #Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_label
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

