import os
import sys
sys.path.append('./lib')
import cv2
import time
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.apmeter import APMeter
from utils import videotransforms
from network.infdetnet import InfDetNet

def evaluate_map(all_scores, all_labels, id_2_class, rm_bg=False):
    '''
    compute mAP for multiple classes
    '''
    ap_list=[]
    start = 0 if not rm_bg else 1
    n_class = all_scores.shape[1]

    for i in range(start, n_class):
        label=id_2_class[i]
        gt_label=[]
        pred_scores=all_scores[:,i]
        for now_label in all_labels:
            if now_label==i:
                gt_label.append(1)
            else:
                gt_label.append(0)

        gt_label = np.asarray(gt_label)
        ap = average_precision_score(gt_label,pred_scores)
        # print(label,ap)
        ap_list.append(ap)
    map=np.nanmean(ap_list)
    # print("mAP:%s" % map)
    return map

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def sigmoid(x):
    return 1/(1+np.exp(-x))

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=str, default='3')
parser.add_argument('-epoch', type=int, default=1000, help='number of max epoch')
parser.add_argument('-gamma', type=float, default=0.7, help='weight of flow loss')
parser.add_argument('-print_freq', type=int, default=100, help='frequency of print training loss')
parser.add_argument('-use_apex', type=bool, default=False, help='use apex to accelerate core computing.')
parser.add_argument('-train', type=str2bool, default=False, help='train or eval')
parser.add_argument('-vis', type=str2bool, default=False, help='visualize flow images')
parser.add_argument('-reload', type=str, default='output/joint_selatt_tgm_infvis')

parser.add_argument('-model_file', type=str, default='output/joint_selatt_tgm')
parser.add_argument('-rgb_model_file', type=str, default='models/rgb_imagenet.pt')
parser.add_argument('-flow_model_file', type=str, default='models/flow_imagenet.pt')
# Uncomment following commands if you use the finetuned model
# parser.add_argument('-rgb_model_file', type=str, default='models/infar_i3d.pt000420.pt')
# parser.add_argument('-flow_model_file', type=str, default='models/flow_i3d.pth')
parser.add_argument('-dataset', type=str, default='infdet')
args = parser.parse_args()
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

device = torch.device('cuda:%s' % args.gpu)
if args.dataset == 'infdet':
    from dataset.infdetdataset import Infdet as Dataset
    from dataset.infdetdataset import mt_collate_fn as collate_fn
    rgb_root = '/raid/chenxu/workspace/inf_det/inf_frames'
    flow_root = '/raid/chenxu/workspace/inf_det/inf_op_frames'
    split_file = '/raid/chenxu/workspace/inf_det/total.json'
    flow_weight = None
    classes = 3
    batch_size = 1
elif args.dataset == 'infar':
    from dataset.infardataset import InfAR as Dataset
    from dataset.infardataset import mt_collate_fn as collate_fn
    from dataset.infardataset import class_id
    rgb_root = '/raid/chenxu/workspace/InfAR/inf_frames'
    flow_root = '/raid/chenxu/workspace/InfAR/inf_op_frames'
    split_file = 'tools/infar/split_2.json'
    flow_weight = args.flow_model_file 
    classes = 12
    id_2_class = {v: k for k, v in class_id.items()}
    batch_size = 2
elif args.dataset == 'infvis':
    from dataset.infvisdataset import InfVis as Dataset
    from dataset.infvisdataset import mt_collate_fn as collate_fn
    from dataset.infvisdataset import class_id
    rgb_root = '/raid/chenxu/workspace/InfVis/inf_frames'
    flow_root = '/raid/chenxu/workspace/InfVis/inf_op_frames'
    split_file = '/raid/chenxu/workspace/InfVis/split.json'
    flow_weight = args.flow_model_file
    classes = 12
    id_2_class = {v: k for k, v in class_id.items()}
    batch_size = 2


if args.use_apex:
    from apex import amp


def load_data(trn_split, test_split):
    # Load Data
    transf = transforms.Compose([videotransforms.CenterCrop(224)])
    if len(trn_split) > 0:
        dataset = Dataset(trn_split, 'training', rgb_root, flow_root,
                          classes=classes, transf=transf)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                 pin_memory=True, collate_fn=collate_fn)
        dataloader.root = rgb_root
    else:
        dataset = None
        dataloader = None

    val_dataset = Dataset(test_split, 'testing', rgb_root, flow_root,
                          classes=classes, transf=transf)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=8,
                                                 pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = rgb_root

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets


def eval_model(model, dataloader):
    results = {}
    t = []
    for data in dataloader:
        other = data[4]
        tic = time.time()
        with torch.no_grad():
            outputs, loss, probs, _ = run_network(model, data, vis=args.vis)
        t.append(time.time()-tic)
        fps = outputs.size()[1]/other[1]
        print(fps)
        results[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], data[3].numpy()[0], fps)
    return results


def train_step(model, optimizer, dataloader):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.

    # Iterate over data.
    for data in dataloader:
        label = data[3]
        optimizer.zero_grad()

        outputs, loss, probs, err = run_network(model, data)

        if outputs is None:
            continue

        error += err.item()
        tot_loss += loss.item()

        if args.use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        #torch.cuda.empty_cache()
        if args.dataset!='infdet':
            if num_iter>0:
                gt = np.concatenate((gt, label.cpu().numpy()), axis=0)
                pred = np.concatenate((pred, probs.cpu().detach().numpy()), axis=0)
            else:
                gt = label.cpu().numpy()
                pred = probs.cpu().detach().numpy()

        num_iter += 1
        if num_iter % args.print_freq == 0:
            print(loss.item())

    optimizer.step()
    optimizer.zero_grad()
    epoch_loss = tot_loss / num_iter
    error = error / num_iter
    if args.dataset=='infdet':
        print('train-{} Loss: {:.4f} mAP: {:.4f}'.format(dataloader.root, epoch_loss, error))
    else:
        map = evaluate_map(pred, gt, id_2_class)
        p_out = np.argmax(pred, axis=1)
        acc = np.sum(gt==p_out)/gt.shape[0]
        print('train-{} Loss: {:.4f} mAP: {:.4f} Acc: {:.4f}'.format(dataloader.root, epoch_loss, map, acc))


def val_step(model, dataloader):
    model.eval()
    apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    full_probs = {}

    # Iterate over data.
    for data in dataloader:
        num_iter += 1
        # other = data[4]
        with torch.no_grad():
            outputs, loss, probs, err = run_network(model, data)
        if outputs is None:
            continue
        apm.add(probs.data.cpu().numpy()[0], data[3].numpy()[0])

        error += err.item()
        tot_loss += loss.item()

        torch.cuda.empty_cache()

    epoch_loss = tot_loss / num_iter
    error = error / num_iter
    map = apm.value().mean()
    print('val-map:', map)
    apm.reset()
    print('val-{} Loss: {:.4f} Acc: {:.4f}'.format(dataloader.root, epoch_loss, error))

    return full_probs, epoch_loss


def val_class_step(model, dataloader):
    model.eval()
    tot_loss = 0.0
    count=0

    for data in dataloader:
        label = data[3]
        with torch.no_grad():
            outputs, loss, probs, err = run_network(model, data)
        tot_loss += loss.item()
        if count:
            gt = np.concatenate((gt, label.cpu().numpy()), axis=0)
            pred = np.concatenate((pred, probs.cpu().detach().numpy()), axis=0)
        else:
            gt = label.cpu().numpy()
            pred = probs.cpu().detach().numpy()
        count+=1

    epoch_loss = tot_loss / count
    map = evaluate_map(pred, gt, id_2_class)
    p_out = np.argmax(pred, axis=1)
    acc = np.sum(gt == p_out) / gt.shape[0]
    print('val-{} Loss: {:.4f} mAP: {:.4f} Acc: {:.4f}'.format(dataloader.root, epoch_loss, map, acc))

    return epoch_loss, acc


def run_network(model, data, vis=False):
    # get the inputs
    inputs, flow_gt, mask, labels, other = data
    b,c,t,h,w = flow_gt.shape

    # wrap them in Variable
    inputs = Variable(inputs.cuda(device=device))
    mask = Variable(mask.cuda(device=device))
    labels = Variable(labels.cuda(device=device))
    flow_gt = Variable(flow_gt.cuda(device=device))

    # forward
    outputs, pred_flow = model([inputs, torch.sum(mask, 1)])

    outputs = outputs.permute(0, 2, 1)
    if args.dataset=='infdet':
        if outputs.shape!=labels.shape:
            labels = labels[:,:-1,:]
            mask = mask[:,:-1]
            # len = outputs.shape[1]
            if outputs.shape!=labels.shape:
                print('%s has been ignored.' % other[0][0])
                print(outputs.shape, labels.shape)
                return None, None, None, None
        probs = torch.sigmoid(outputs) * mask.unsqueeze(2)
    else:
        if outputs.shape[1]!=mask.shape[1]:
            mask = mask[:,:-1]
            # len = outputs.shape[1]
            if outputs.shape[1]!=mask.shape[1]:
                print(outputs.shape, mask.shape)
                return None, None, None, None
        probs = torch.sigmoid(outputs) * mask.unsqueeze(2)
        probs = torch.mean(probs, dim=1)
        ones = torch.sparse.torch.eye(classes)
        one_hot = ones.cuda(device=device).index_select(0, labels)
        labels = one_hot.unsqueeze(dim=1).repeat(1, outputs.shape[1], 1)

    # binary action-prediction loss
    loss_c = F.binary_cross_entropy_with_logits(outputs, labels, reduction='sum')
    loss_c = torch.sum(loss_c) / torch.sum(mask)  # mean over valid entries
    loss_f = F.l1_loss(flow_gt, pred_flow, reduction='sum')
    loss_f = torch.sum(loss_f) / (torch.sum(mask)*c*h*w)
    loss = loss_c+loss_f*args.gamma

    # Visualize optical flow
    if vis:
        v_t = random.randrange(t-1)
        f_g = flow_gt.squeeze(0)[:, v_t, ...].permute(1,2,0)
        f_g = ((f_g + 1) / 2 * 255).cpu().numpy().astype(np.uint8)

        f_p = pred_flow.squeeze(0)[:, v_t, ...].permute(1,2,0)
        f_p = ((f_p + 1) / 2 * 255).cpu().numpy().astype(np.uint8)

        print('Saving flow imgs of %s...' % other[0][0])
        cv2.imwrite('output/flow_imgs/%s_%d_gt.jpg'%(other[0][0],v_t), f_g)
        cv2.imwrite('output/flow_imgs/%s_%d_pr.jpg'%(other[0][0],v_t), f_p)
        cv2.destroyAllWindows()

    f_g = ((flow_gt + 1) / 2 * 255)
    f_p = ((pred_flow + 1) / 2 * 255)
    p_error = torch.sqrt((f_g-f_p)*(f_g-f_p)).mean()

    return outputs, loss, probs, p_error


def load_exist_weight(model, w_dict):
    m_dict = model.state_dict()
    for m, v in w_dict.items():
        if 'detector' in m or 'flow_est.flow_pred' in m or\
                m=='flow_feat_net.Conv3d_1a_7x7.conv3d.weight':
            continue
        m_dict[m]=v
    model.load_state_dict(m_dict)


def run():
    model = InfDetNet('sel_cross_attention',
                      rgb_weight=args.rgb_model_file,
                      flow_weight=flow_weight,
                      classes=classes, device=device)

    if args.train:
        print('Training...')
        start_epoch = 0
        dataloaders, datasets = load_data(split_file, split_file)
        lr = 6*0.01*batch_size/len(datasets['train'])
        print(lr)
        if args.reload:
            model.load_state_dict(torch.load(args.reload))
            # load_exist_weight(model, torch.load(args.reload))
            start_epoch = int(args.reload.split('_')[-1])+1
            print('Training from epoch %d.' % start_epoch)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=5e-4)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
        since_time = time.time()
        best_loss = 1000
        best_ap = 0.0
        if args.use_apex:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        for epoch in range(start_epoch, args.epoch):
            print('Epoch {}/{}'.format(epoch, args.epoch - 1))
            print('-' * 10)
            start = time.time()

            probs = []
            train_step(model, optimizer, dataloaders['train'])
            if args.dataset == 'infdet':
                prob_val, val_loss = val_step(model, dataloaders['val'])
                probs.append(prob_val)
                lr_sched.step(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), '{}_{}'.format(args.model_file+'_'+args.dataset, epoch))
            else:
                val_loss, val_ap = val_class_step(model, dataloaders['val'])
                lr_sched.step(val_loss)
                if val_ap > best_ap:
                    best_ap = val_ap
                    torch.save(model.state_dict(), '{}_{}'.format(args.model_file+'_'+args.dataset, epoch))

            print('time elapsed: %.2f s'%(time.time()-start))
        print('Training finished in %ds.' % (time.time()-since_time))
    else:
        print('Evaluating...')
        mdl = torch.load('{}_{}'.format(args.model_file+'_'+args.dataset, '43'))
        model.load_state_dict(mdl)
        model.eval()
        dataloaders, datasets = load_data('', split_file)

        rgb_results = eval_model(model, dataloaders['val'])

        apm = APMeter()
        preds = {}
        for vid in rgb_results.keys():
            o,p,l,fps = rgb_results[vid]
            apm.add(sigmoid(o), l)
            preds[vid] = (sigmoid(o)[:,:20].tolist(), fps.tolist())
        print ('MAP:', apm.value().mean().numpy())
        with open('preds.json', 'w') as out:
            json.dump(preds, out)


if __name__ == '__main__':
    run()
