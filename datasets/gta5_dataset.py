import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
from torch.utils import data
import cv2
from PIL import Image


class GTA5DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, img_size=(321, 321), norm=False, random_mirror=False, random_crop=False, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.img_size = img_size
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.norm = norm
        self.random_mirror = random_mirror
        self.random_crop = random_crop
        if not max_iters == None:
            self.img_ids = self.img_ids * \
                int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "lbl": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]
        image = cv2.imread(datafiles["img"], 1)
        label = cv2.imread(datafiles["lbl"], 0)
        # resize
        image = Image.fromarray(image)
        label = Image.fromarray(label)
        if self.random_crop:
            img_w, img_h = image.size
            crop_w, crop_h = self.img_size
            if img_h < crop_h or img_w < crop_w:
                image = image.resize(self.img_size, Image.BICUBIC)
                label = label.resize(self.img_size, Image.NEAREST)
            else:
                h_off = random.randint(0, img_h - crop_h)
                w_off = random.randint(0, img_w - crop_w)
                image = image.crop((w_off, h_off, w_off+crop_w, h_off+crop_h))
                label = label.crop((w_off, h_off, w_off+crop_w, h_off+crop_h))
        else:
            image = image.resize(self.img_size, Image.BICUBIC)
            label = label.resize(self.img_size, Image.NEAREST)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape[:2]
        if self.random_mirror:
            flip = np.random.choice(2)*2 - 1
            image = image[:, ::flip, :]
            label_copy = label_copy[:, ::flip]
        if self.norm:
            image = image / 255.0
            image -= np.array([0.485, 0.456, 0.406])
            image = image / np.array([0.229, 0.224, 0.225])
        else:
            image = image - np.array([122.67892, 116.66877, 104.00699])
        image = image.transpose((2, 0, 1)).astype(np.float32)
        return image.copy(), label_copy.copy(), np.array(size), name
