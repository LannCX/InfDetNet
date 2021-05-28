import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import os
import cv2
import json
import random
import numpy as np

class_id = {'skip':0,
            'handshake': 1,
            'push': 2,
            'punch': 3,
            'wave2': 4,
            'jump': 5,
            'hug': 6,
            'jog': 7,
            'walk': 8,
            'handclapping': 9,
            'wave1': 10,
            'fight': 11}

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(image_dir, vid):
  frames = []
  nf = len(os.listdir(os.path.join(image_dir, vid)))
  for i in range(9, nf-7):
    img = cv2.imread(os.path.join(image_dir, vid, str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
    w,h,c = img.shape
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    img = (img/255.)*2 - 1
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid):
    frames = []
    nf = len(os.listdir(os.path.join(image_dir, vid)))//2
    for i in range(9, nf-7):
        imgx = cv2.imread(os.path.join(image_dir, vid, 'flow_x_' + str(i).zfill(6) + '.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, 'flow_y_' + str(i).zfill(6) + '.jpg'), cv2.IMREAD_GRAYSCALE)

        w, h = imgx.shape
        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


class InfAR(data_utl.Dataset):
    def __init__(self, split_file, split, rgb_root, flow_root, classes, transf=None, down_ratio=4):
        with open(split_file) as f:
            split_dict = json.load(f)
            self.vid_list = split_dict[split]
        self.split_file = split_file
        self.transf = transf
        self.rgb_root = rgb_root
        self.flow_root = flow_root
        self.down_ratio = down_ratio
        self.in_mem = {}

    def __getitem__(self, index):
        vid = self.vid_list[index]
        label = class_id[vid.split('_')[-1]]
        imgs = load_rgb_frames(self.rgb_root, vid)
        imgs = self.transf(imgs)

        flow_gt = load_flow_frames(self.flow_root, vid)
        flow_gt = self.transf(flow_gt)
        flow_gt = flow_gt[:, ::self.down_ratio, ::self.down_ratio, :]

        return imgs, flow_gt, label, [vid]

    def __len__(self):
        return len(self.vid_list)


def mt_collate_fn(batch):
    "Pads data and puts it into a tensor of same dimensions"
    max_len = 0
    for b in batch:
        if b[0].shape[0] > max_len:
            max_len = b[0].shape[0]
    max_feat_len = max_len//8

    new_batch = []
    for b in batch:
        f_x = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]), np.float32)
        f_y = np.zeros((max_len, b[1].shape[1], b[1].shape[2], b[1].shape[3]), np.float32)
        m = np.zeros(max_feat_len, np.float32)
        f_x[:b[0].shape[0]] = b[0]
        f_y[:b[1].shape[0]] = b[1]
        m[:b[0].shape[0]] = 1
        new_batch.append([video_to_tensor(f_x), video_to_tensor(f_y), torch.from_numpy(m), b[2], b[3]])

    return default_collate(new_batch)

