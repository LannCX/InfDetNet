import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import temporal_structure_filter as tsf
from tgm import TGM
from attention_layer import *

class SuperEvent(nn.Module):
    def __init__(self, classes=65):
        super(SuperEvent, self).__init__()

        self.classes = classes
        self.dropout = nn.Dropout(0.7)
        self.add_module('d', self.dropout)

        self.super_event = tsf.TSF(3)#, False)
        self.add_module('sup', self.super_event)
        self.super_event2 = tsf.TSF(3)#, False)
        self.add_module('sup2', self.super_event2)


        # we have 2xD*3
        # we want to learn a per-class weighting
        # to take 2xD*3 to D*3
        self.cls_wts = nn.Parameter(torch.Tensor(classes))
        
        self.sup_mat = nn.Parameter(torch.Tensor(1, classes, 1024))
        stdv = 1./np.sqrt(1024+1024*3)
        # self.cls_wts.data.uniform_(-stdv, stdv)
        self.sup_mat.data.uniform_(-stdv, stdv)

        self.per_frame = nn.Conv3d(1024, classes, (1,1,1))
        self.per_frame.weight.data.uniform_(-stdv, stdv)
        self.per_frame.bias.data.uniform_(-stdv, stdv)
        self.add_module('pf', self.per_frame)
        
    def forward(self, inp):
        inp[0] = self.dropout(inp[0])
        val = False
        dim = 1
        if inp[0].size()[0] == 1:
            val = True
            dim = 0

        super_event = self.dropout(torch.stack([self.super_event(inp).squeeze(), self.super_event2(inp).squeeze()], dim=dim))
        if val:
            super_event = super_event.unsqueeze(0)
        # we have B x 2 x D*3
        # we want B x C x D*3

        # now we have C x 2 matrix
        cls_wts = torch.stack([torch.sigmoid(self.cls_wts), 1-torch.sigmoid(self.cls_wts)], dim=1)

        # now we do a bmm to get B x C x D*3
        super_event = torch.bmm(cls_wts.expand(inp[0].size()[0], -1, -1), super_event)
        del cls_wts

        # apply the super-event weights
        super_event = torch.sum(self.sup_mat * super_event, dim=2)
        #super_event = self.sup_mat(super_event.view(-1, 1024)).view(-1, self.classes)
        
        super_event = super_event.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        cls = self.per_frame(inp[0])
        return super_event+cls


def compute_pad(stride, k, s):
    if s % stride == 0:
        return max(k - stride, 0)
    else:
        return max(k - (s % stride), 0)


class SubConv(tsf.TSF):
    """
    Subevents as temporal conv
    """
    def __init__(self, inp, num_f,  length):
        super(SubConv, self).__init__(num_f)
        
        self.inp = inp
        self.length = length

    
    def forward(self, x):
        # overwrite the forward pass to get the TSF as conv kernels
        t = x.size(2)
        k = super(SubConv, self).get_filters(torch.tanh(self.delta), torch.tanh(self.gamma), torch.tanh(self.center), self.length, self.length)
        k = k.squeeze().unsqueeze(1).unsqueeze(1)#.repeat(1, 1, self.inp, 1)
        p = compute_pad(1, self.length, t)
        pad_f = p // 2
        pad_b = p - pad_f
        x = F.pad(x, (pad_f, pad_b)).unsqueeze(1)
        return F.conv2d(x, k).squeeze(1)


class SubConv2(tsf.TSF):
    """
    Subevents as temporal conv
    """
    def __init__(self, inp, num_f,  length, c=1):
        super(SubConv2, self).__init__(num_f)
        
        self.inp = inp
        self.length = length
        self.c = c
        
        self.soft_attn = nn.Parameter(torch.Tensor(c, num_f))
    
    def forward(self, x):
        # overwrite the forward pass to get the TSF as conv kernels
        t = x.size(2)
        k = super(SubConv2, self).get_filters(torch.tanh(self.delta), torch.tanh(self.gamma), torch.tanh(self.center), self.length, self.length)
        # k is shape 1xNxL
        k = k.squeeze()
        # is k now NxL
        # apply soft attention to conver (CxN)*(NxL) to CxL

        # make attn sum to 1 along the num_gaussians
        soft_attn = F.softmax(self.soft_attn, dim=1)
        #print soft_attn
        k = torch.mm(soft_attn, k)

        # make k Cx1x1xL
        k = k.unsqueeze(1).unsqueeze(1)
        p = compute_pad(1, self.length, t)
        pad_f = p // 2
        pad_b = p - pad_f

        # x is shape CxDxT
        x = F.pad(x, (pad_f, pad_b))
        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        if x.size(1) == 1:
            x = x.expand(-1, self.c, -1, -1)
        #print x.size(), k.size(), self.c
        # use groups to separate the class channels
        return F.conv2d(x, k, groups=self.c).squeeze(1)
    

class StackSub(nn.Module):

    def __init__(self, inp, num_f, length, classes):
        super(StackSub, self).__init__()

        # each return T*num_f
        self.sub_event1 = SubConv(inp, num_f, length)
        self.sub_event2 = SubConv(inp, num_f, length)
        self.sub_event3 = SubConv(inp, num_f, length)
        
        self.classify = nn.Conv1d(num_f*inp, classes, 1)
        self.dropout = nn.Dropout()
        
        self.inp = inp
        self.num_f = num_f
        self.classes = classes

    def forward(self, inp):
        val = False
        dim = 1
        f = inp[0].squeeze()
        if inp[0].size()[0] == 1:
            val = True
            dim = 0
            f = f.unsqueeze(0)

        sub_event = torch.max(F.relu(self.sub_event1(f)), dim=1)[0]
        sub_event = torch.max(F.relu(self.sub_event2(sub_event)), dim=1)[0]
        sub_event = self.sub_event3(sub_event)
            
        sub_event = self.dropout(sub_event).view(-1, self.num_f*self.inp, f.size(2))
        cls = F.relu(sub_event)
        return self.classify(cls)


class Hierarchy(nn.Module):
    def __init__(self, inp, classes=8):
        super(Hierarchy, self).__init__()

        self.classes = classes
        self.dropout = nn.Dropout(0)
        self.add_module('d', self.dropout)

        self.super_event = tsf.TSF(3)
        self.add_module('sup', self.super_event)
        self.super_event2 = tsf.TSF(3)
        self.add_module('sup2', self.super_event2)

        # we have 2xD
        # we want to learn a per-class weighting
        # to take 2xD to D
        self.cls_wts = nn.Parameter(torch.Tensor(classes))
        
        self.sup_mat = nn.Parameter(torch.Tensor(1, classes, inp))
        stdv = 1./np.sqrt(inp+inp)
        self.sup_mat.data.uniform_(-stdv, stdv)

        self.sub_event1 = TGM(inp, 16, 5, c_in=1, c_out=8, soft=False)
        self.sub_event2 = TGM(inp, 16, 5, c_in=8, c_out=8, soft=False)
        self.sub_event3 = TGM(inp, 16, 5, c_in=8, c_out=8, soft=False)

        self.h = nn.Conv1d(inp+1*inp+classes, 512, 1)
        self.classify = nn.Conv1d(512, classes, 1)
        self.inp = inp
        
    def forward(self, inp):
        val = False
        dim = 1
        if inp[0].size()[0] == 1:
            val = True
            dim = 0

        super_event = torch.stack([self.super_event(inp).squeeze(), self.super_event2(inp).squeeze()], dim=dim)
        f = inp[0].squeeze()
        if val:
            super_event = super_event.unsqueeze(0)
            f = f.unsqueeze(0)
        # we have B x 2 x D
        # we want B x C x D

        # now we have C x 2 matrix
        cls_wts = torch.stack([torch.sigmoid(self.cls_wts), 1-torch.sigmoid(self.cls_wts)], dim=1)

        # now we do a bmm to get B x C x D*3
        super_event = torch.bmm(cls_wts.expand(inp[0].size()[0], -1, -1), super_event)
        del cls_wts

        # apply the super-event weights
        super_event = torch.sum(self.sup_mat * super_event, dim=2)
        
        super_event = self.dropout(super_event).view(-1,self.classes,1).expand(-1,self.classes,f.size(2))

        sub_event = self.sub_event1(f)
        sub_event = self.sub_event2(sub_event)
        sub_event = self.dropout(torch.max(self.sub_event3(sub_event), dim=1)[0])

        cls = F.relu(torch.cat([self.dropout(f), sub_event, super_event], dim=1))
        cls = F.relu(self.h(cls))
        return self.classify(cls)


class AttHierarchy(nn.Module):
    def __init__(self, inp, att_layer, classes=8):
        super(AttHierarchy, self).__init__()

        self.classes = classes
        self.dropout = nn.Dropout(0)
        self.add_module('d', self.dropout)

        self.super_event = tsf.TSF(3)
        self.add_module('sup', self.super_event)
        self.super_event2 = tsf.TSF(3)
        self.add_module('sup2', self.super_event2)

        if att_layer=='self_attention':
            self.att_module = SelfAttentionLayer(inp)
            self.one_ch = True
        elif att_layer=='cross_attention':
            self.att_module = CrossAttentinLayer(inp)
            self.one_ch = False
        elif att_layer=='sel_cross_attention':
            self.att_module = SelectAttentionLayer(inp)
            self.one_ch = False
        else:
            raise(KeyError, 'No such attention layer named:%s' % att_layer)

        # we have 2xD
        # we want to learn a per-class weighting
        # to take 2xD to D
        self.cls_wts = nn.Parameter(torch.Tensor(classes))

        self.sup_mat = nn.Parameter(torch.Tensor(1, classes, inp))
        stdv = 1. / np.sqrt(inp + inp)
        self.sup_mat.data.uniform_(-stdv, stdv)

        self.sub_event1 = TGM(inp, 16, 5, c_in=1, c_out=8, soft=False)
        self.sub_event2 = TGM(inp, 16, 5, c_in=8, c_out=8, soft=False)
        self.sub_event3 = TGM(inp, 16, 5, c_in=8, c_out=8, soft=False)

        self.h = nn.Conv1d(inp + 1 * inp + classes, 512, 1)
        self.classify = nn.Conv1d(512, classes, 1)
        self.inp = inp

    def forward(self, inp):
        val = False
        dim = 1
        if inp[0].size()[0] == 1:
            val = True
            dim = 0

        if self.one_ch:
            proj_inp, attention = self.att_module(inp[0])
            new_inp = [proj_inp]+inp[1:]
        else:
            proj_inp, attention = self.att_module(inp[0], inp[1])
            new_inp = [proj_inp]+inp[2:]

        super_event = torch.stack([self.super_event(new_inp).squeeze(), self.super_event2(new_inp).squeeze()], dim=dim)
        f = inp[0].squeeze()
        if val:
            super_event = super_event.unsqueeze(0)
            f = f.unsqueeze(0)
        # we have B x 2 x D
        # we want B x C x D

        # now we have C x 2 matrix
        cls_wts = torch.stack([torch.sigmoid(self.cls_wts), 1 - torch.sigmoid(self.cls_wts)], dim=1)

        # now we do a bmm to get B x C x D*3
        super_event = torch.bmm(cls_wts.expand(inp[0].size()[0], -1, -1), super_event)
        del cls_wts

        # apply the super-event weights
        super_event = torch.sum(self.sup_mat * super_event, dim=2)

        super_event = self.dropout(super_event).view(-1, self.classes, 1).expand(-1, self.classes, f.size(2))

        sub_event = self.sub_event1(f)
        sub_event = self.sub_event2(sub_event)
        sub_event = self.dropout(torch.max(self.sub_event3(sub_event), dim=1)[0])

        cls = F.relu(torch.cat([self.dropout(f), sub_event, super_event], dim=1))
        cls = F.relu(self.h(cls))
        return self.classify(cls)


class StackTGM(nn.Module):
    def __init__(self, inp, classes=8):
        super(StackTGM, self).__init__()

        self.classes = classes
        self.dropout = nn.Dropout()
        self.add_module('d', self.dropout)

        self.sub_event1 = TGM(inp, 16, 5, c_in=1, c_out=8, soft=False)
        self.sub_event2 = TGM(inp, 16, 5, c_in=8, c_out=8, soft=False)
        self.sub_event3 = TGM(inp, 16, 5, c_in=8, c_out=8, soft=False)

        self.h = nn.Conv1d(inp+1*inp, 512, 1)
        self.classify = nn.Conv1d(512, classes, 1)
        self.inp = inp
        
    def forward(self, inp):
        val = False
        dim = 1
        if inp[0].size()[0] == 1:
            val = True
            dim = 0

        f = inp[0].squeeze()
        if val:
            f = f.unsqueeze(0)

        sub_event = self.sub_event1(f)
        sub_event = self.sub_event2(sub_event)
        sub_event = self.sub_event3(sub_event)
        sub_event = self.dropout(torch.max(sub_event, dim=1)[0])

        cls = F.relu(torch.cat([self.dropout(f), sub_event], dim=1))
        cls = F.relu(self.h(cls))
        return self.classify(cls)


def get_baseline_model(classes=65):
    model = nn.Sequential(
#        nn.Dropout(0.5),
        nn.Conv1d(1024,1024,3, padding=1),
        nn.ReLU(),
        nn.Conv1d(1024, classes, 1, padding=0))
    return model.cuda()


def get_super_model(gpu, classes=65):
    model = SuperEvent(classes)
    return model.cuda()


def get_hier(classes):
    model = Hierarchy(1024*2, classes)
    model.cuda()
    return model


def get_tgm(classes):
    model = StackTGM(1024, classes)
    model.cuda()
    return model


def get_att_hier(classes, att_name):
    model = AttHierarchy(1024, att_name, classes)
    model.cuda()
    return model
