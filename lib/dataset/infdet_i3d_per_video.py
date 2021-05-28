import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py

import os
import os.path

joint = False


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


def make_dataset(split_file, split, root, num_classes=3):
    global joint
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    if isinstance(root, list):
        joint = True
        x_root, y_root = root
        for vid in data.keys():
            if data[vid]['subset'] != split:
                continue

            if not os.path.exists(os.path.join(x_root, vid+'.npy')) or \
                    not os.path.exists(os.path.join(y_root, vid+'.npy')):
                continue
            x_fts = np.load(os.path.join(x_root, vid+'.npy'))
            y_fts = np.load(os.path.join(y_root, vid+'.npy'))
            assert x_fts.shape[0]==y_fts.shape[0]
            num_feat = x_fts.shape[0]
            label = np.zeros((num_feat, num_classes), np.float32)

            fps = num_feat / data[vid]['duration']
            for ann in data[vid]['actions']:
                for fr in range(0, num_feat, 1):
                    if ann[1] < fr / fps < ann[2]:
                        label[fr, ann[0] - 1] = 1  # binary classification
            dataset.append((vid, label, data[vid]['duration']))
            i += 1
    elif isinstance(root, str):
        joint = False
        for vid in data.keys():
            if data[vid]['subset'] != split:
                continue

            if not os.path.exists(os.path.join(root, vid + '.npy')):
                continue
            fts = np.load(os.path.join(root, vid + '.npy'))
            num_feat = fts.shape[0]
            label = np.zeros((num_feat, num_classes), np.float32)

            fps = num_feat/data[vid]['duration']
            for ann in data[vid]['actions']:
                for fr in range(0, num_feat, 1):
                    if ann[1] < fr/fps < ann[2]:
                        label[fr, ann[0] - 1] = 1  # binary classification
            dataset.append((vid, label, data[vid]['duration']))
            i += 1

    return dataset


class MultiThumos(data_utl.Dataset):

    def __init__(self, split_file, split, root, batch_size):

        self.data = make_dataset(split_file, split, root)
        self.split_file = split_file
        self.batch_size = batch_size
        self.root = root
        self.in_mem = {}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        entry = self.data[index]
        label = entry[1]
        if isinstance(self.root, str):
            if entry[0] in self.in_mem:
                feat = self.in_mem[entry[0]]
            else:
                feat = np.load(os.path.join(self.root, entry[0] + '.npy'))
                feat = feat.reshape((feat.shape[0], 1, 1, 1024))
                feat = feat.astype(np.float32)
            return feat, label, [entry[0], entry[2]]
        elif isinstance(self.root, list):
            if entry[0] in self.in_mem:
                feat_x, feat_y = self.in_mem[entry[0]]
            else:
                feat_x = np.load(os.path.join(self.root[0], entry[0] + '.npy'))
                feat_y = np.load(os.path.join(self.root[1], entry[0] + '.npy'))

                feat_x = feat_x.reshape((feat_x.shape[0], 1, 1, 1024)).astype(np.float32)
                feat_y = feat_y.reshape((feat_y.shape[0], 1, 1, 1024)).astype(np.float32)
            return feat_x, feat_y, label, [entry[0], entry[2]]

    def __len__(self):
        return len(self.data)


def mt_collate_fn(batch):
    "Pads data and puts it into a tensor of same dimensions"
    max_len = 0
    for b in batch:
        if b[0].shape[0] > max_len:
            max_len = b[0].shape[0]

    new_batch = []
    if joint:
        for b in batch:
            f_x = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]), np.float32)
            f_y = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[1].shape[3]), np.float32)
            m = np.zeros((max_len), np.float32)
            l = np.zeros((max_len, b[2].shape[1]), np.float32)
            f_x[:b[0].shape[0]] = b[0]
            f_y[:b[1].shape[0]] = b[1]
            m[:b[0].shape[0]] = 1
            l[:b[0].shape[0], :] = b[2]
            new_batch.append([video_to_tensor(f_x), video_to_tensor(f_y), torch.from_numpy(m), torch.from_numpy(l), b[3]])
    else:
        for b in batch:
            f = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]), np.float32)
            m = np.zeros((max_len), np.float32)
            l = np.zeros((max_len, b[1].shape[1]), np.float32)
            f[:b[0].shape[0]] = b[0]
            m[:b[0].shape[0]] = 1
            l[:b[0].shape[0], :] = b[1]
            new_batch.append([video_to_tensor(f), torch.from_numpy(m), torch.from_numpy(l), b[2]])

    return default_collate(new_batch)

