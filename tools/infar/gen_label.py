import os
import cv2
import time
import json
import random


def gen_label(root='/raid/chenxu/workspace/InfVis/inf_frames'):
    class_label = {}
    gt = {}
    count = 0
    for root, dirs, files in os.walk(root):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            act = dir_name.split('_')[-1]
            if act not in class_label:
                class_label[act] = count
                count+=1
            length = len(os.listdir(dir_path)) / 25
            tmp = {}
            tmp['duration'] = length
            tmp['subset'] = 'training'
            tmp['actions'] = [[class_label[act], 0, length]]
            gt[dir_name] = tmp

    with open('infvis_mapping', 'w') as f:
        for k,v in class_label.items():
            f.write('%s %s %s\n'%(v,k,v))


def gen_split(root='/raid/chenxu/workspace/InfAR/inf'):
    split_dict={'training':[], 'testing':[]}
    for k in range(5):
        for root, dirs, _ in os.walk(root):
            for dir_name in dirs:
                files = [x.split('.')[0] for x in os.listdir(os.path.join(root,dir_name))]
                n_p = int(len(files)*0.6)
                trn_files = random.sample(files, n_p)
                split_dict['training'].extend(trn_files)
                tst_files = set(files).difference(trn_files)
                split_dict['testing'].extend(tst_files)
        assert len(split_dict['training'])+len(split_dict['testing'])==600
        with open('split_%d.json'%k, 'w') as f:
            json.dump(split_dict, f, indent=2)


def gen_gt(src_root='/raid/chenxu/workspace/InfAR/inf_frames'):
    class_label = {}
    with open('infar_mapping', 'r') as f:
        for line in f:
            e = line.split()
            class_label[e[1]] = int(e[0])

    for k in range(5):
        with open('split_%d.json'%k, 'r') as f:
            split_dict = json.load(f)
        gt = {}
        for root, dirs, files in os.walk(src_root):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                act = dir_name.split('_')[-1]
                length = len(os.listdir(dir_path)) / 25
                tmp = {}
                tmp['duration'] = length
                tmp['subset'] = 'training' if dir_name in split_dict['train'] else 'testing'
                tmp['actions'] = [[class_label[act], 0, length]]
                gt[dir_name] = tmp
        assert len(gt)==600
        with open('infar_instance_%d.json'%k, 'w') as f:
            json.dump(gt, f, indent=2)


def gen_gt_1(src_root='/raid/chenxu/workspace/InfVis/inf_frames'):
    class_label = {}
    with open('infvis_mapping', 'r') as f:
        for line in f:
            e = line.split()
            class_label[e[1]] = int(e[0])

    with open('/raid/chenxu/workspace/InfVis/split.json', 'r') as f:
        split_dict = json.load(f)
    gt = {}
    for root, dirs, files in os.walk(src_root):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            act = dir_name.split('_')[-1]
            length = len(os.listdir(dir_path)) / 25
            tmp = {}
            tmp['duration'] = length
            tmp['subset'] = 'training' if dir_name in split_dict['train'] else 'testing'
            tmp['actions'] = [[class_label[act], 0, length]]
            gt[dir_name] = tmp
    assert len(gt)==1200
    with open('infvis_instance.json', 'w') as f:
        json.dump(gt, f, indent=2)

if __name__ == '__main__':
    # gen_label()
    gen_gt_1()
