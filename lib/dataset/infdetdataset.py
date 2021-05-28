import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import os
import cv2
import json
import random
import numpy as np


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


def load_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start+num):
        img = cv2.imread(os.path.join(image_dir, vid, str(i).zfill(5)+'.jpg'))[:, :, [2, 1, 0]]
        w,h,c = img.shape
        if w < 226 or h < 226:
            d = 226.-min(w,h)
            sc = 1+d/min(w,h)
            img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
        img = (img/255.)*2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, num_classes=157):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        if data[vid]['subset'] != split:
            continue
        if not os.path.exists(os.path.join(root, vid)):
            continue

        num_frames = len(os.listdir(os.path.join(root, vid)))
        if num_frames < 66:
            continue

        feat_len = num_frames//8
        label = np.zeros((feat_len, num_classes), np.float32)
        fps = feat_len / data[vid]['duration']

        for ann in data[vid]['actions']:
            for fr in range(0, num_frames, 1):
                if ann[1] < fr/fps < ann[2]:
                    label[fr, ann[0]-1] = 1  # binary classification
        dataset.append((vid, label, data[vid]['duration'], num_frames))
        i += 1

    return dataset


class Infdet(data_utl.Dataset):

    def __init__(self, split_file, split, rgb_root, flow_root, classes, transf=None, down_ratio=4):
        self.data = make_dataset(split_file, split, rgb_root, num_classes=classes)
        self.split_file = split_file
        self.transf = transf
        self.rgb_root = rgb_root
        self.flow_root = flow_root
        self.down_ratio = down_ratio
        self.in_mem = {}

    def __getitem__(self, index):
        vid, label, dur, nf = self.data[index]

        imgs = load_frames(self.rgb_root, vid, 0, nf)
        imgs = self.transf(imgs)

        flow_gt = load_frames(self.flow_root, vid, 0, nf)
        flow_gt = self.transf(flow_gt)
        flow_gt = flow_gt[:, ::self.down_ratio,  ::self.down_ratio, :]

        return imgs, flow_gt, torch.from_numpy(label), [vid, dur]

    def __len__(self):
        return len(self.data)


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
        l = np.zeros((max_feat_len, b[2].shape[1]), np.float32)
        f_x[:b[0].shape[0]] = b[0]
        f_y[:b[1].shape[0]] = b[1]
        m[:b[2].shape[0]] = 1
        l[:b[2].shape[0], :] = b[2]
        new_batch.append([video_to_tensor(f_x), video_to_tensor(f_y), torch.from_numpy(m), torch.from_numpy(l), b[3]])

    return default_collate(new_batch)

