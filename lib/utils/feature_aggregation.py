import os
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from tqdm import tqdm

ROOT = '/raid/chenxu/workspace/inf_det'
RGB_I3D_PATH = os.path.join(ROOT, 'inf_i3d')
FLOW_I3D_PATH = os.path.join(ROOT, 'inf_op_i3d')


def softmax(x, dim=1):
    axis_max = x.max(axis=dim)
    x = x-axis_max.reshape(-1,1)

    exp = np.exp(x)
    sum = np.sum(exp, axis=dim, keepdims=True)
    score = exp/sum
    return score


def self_attention(gamma=0.5):
    save_dir = os.path.join(ROOT, 'rgb_atten_i3d')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_files = os.listdir(RGB_I3D_PATH)
    pbar = tqdm(total=len(all_files))
    for file in all_files:
        pbar.update(1)
        fts = np.load(os.path.join(RGB_I3D_PATH, file))
        inf_fts = fts.squeeze()
        # fts = np.load(os.path.join(FLOW_I3D_PATH, file))
        # op_fts = fts.squeeze()
        energy = np.matmul(inf_fts, np.transpose(inf_fts))
        attention = softmax(energy)
        out = np.matmul(attention, inf_fts)
        out_ft = gamma*out+inf_fts
        out_ft = out_ft[:, np.newaxis, np.newaxis, :]
        save_path = os.path.join(save_dir, file)
        np.save(save_path, out_ft)
    pbar.close()


def simple_fuse(mode='concat'):
    save_dir = os.path.join(ROOT, 'con_%s_i3d' % mode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_files = os.listdir(RGB_I3D_PATH)
    pbar = tqdm(total=len(all_files))
    for file in all_files:
        pbar.update(1)
        fts = np.load(os.path.join(RGB_I3D_PATH, file))
        inf_fts = fts.squeeze()
        fts = np.load(os.path.join(FLOW_I3D_PATH, file))
        op_fts = fts.squeeze()

        if mode=='min':
            out_ft = np.minimum(inf_fts, op_fts)
        elif mode=='max':
            out_ft = np.maximum(inf_fts, op_fts)
        elif mode=='concat':
            out_ft = np.concatenate((inf_fts, op_fts),axis=1)
        else:
            raise(KeyError, 'Not supported mode %s' % mode)
        out_ft = out_ft[:, np.newaxis, np.newaxis, :]
        save_path = os.path.join(save_dir, file)
        np.save(save_path, out_ft)
    pbar.close()


if __name__ == '__main__':
    simple_fuse('concat')
