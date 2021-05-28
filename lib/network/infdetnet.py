import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

from collections import namedtuple
import os
import time
import numpy as np
import network.temporal_structure_filter as tsf
from network.tgm import TGM
from network.i3d import InceptionI3d as I3D
from network.attention_layer import *
from torchvision.models.resnet import conv3x3

# Test FlowNet
from network.submodules import *
# from .correlation_package.correlation import Correlation

device_1 = torch.device('cuda:0')
device_2 = torch.device('cuda:1')
device_3 = torch.device('cuda:2')
device_4 = torch.device('cuda:3')


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Encoder(nn.Module):

    def __init__(self, c_in, block, groups=1, width_per_group=64, norm_layer=None):
        super(Encoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(c_in, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, 1)
        self.layer2 = self._make_layer(block, 128, 1, stride=2)
        self.layer3 = self._make_layer(block, 256, 1)
        self.layer4 = self._make_layer(block, 256, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class OptEstNet(nn.Module):

    def __init__(self, c_in, c_out):
        super(OptEstNet, self).__init__()

        # Encoder
        self.encoder = Encoder(c_in, BasicBlock, norm_layer=nn.InstanceNorm2d)

        # Decoder
        self.deconv4 = self.deconv(256, 128)
        self.deconv3 = self.deconv(128, 64)
        self.deconv2 = self.deconv(64, 32)
        # self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')

        self.flow_pred = nn.Conv2d(32, c_out, kernel_size=1)

    def deconv(self, in_planes, out_planes):
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        # x = x.to(device_2)
        x = self.encoder(x)
        # x = x.to(device_4)

        x = self.deconv4(x)
        x = self.deconv3(x)
        x = self.deconv2(x)
        # x = self.upsample(x)
        x = self.flow_pred(x)
        x = torch.tanh(x)

        return x


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

        super_event = torch.stack([self.super_event(inp).squeeze(), self.super_event2(inp).squeeze()], dim=dim)
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


class InfDetNet(nn.Module):
    def __init__(self, atten_type, rgb_weight, flow_weight=None, classes=8, c_in=1024, device=None):
        super(InfDetNet, self).__init__()
        if device is not None:
            self.default_device = device
        else:
            self.default_device = device_1

        # Basic configuration
        self.inf_3dnet_weight = rgb_weight
        self.flow_3dnet_weight = flow_weight

        if flow_weight is not None:
            self.flow_est = OptEstNet(6, 2)
            self.flow_feat_net = I3D(400, in_channels=2)
            self.flow_feat_net.load_state_dict(torch.load(self.flow_3dnet_weight))
        else:
            self.flow_est = OptEstNet(6, 3)
            self.flow_feat_net = I3D(12, in_channels=3)
            self.flow_feat_net.load_state_dict(torch.load(self.inf_3dnet_weight))

        self.inf_feat_net = I3D(12, in_channels=3)
        if atten_type=='self_attention':
            self.cmam = SelfAttentionLayer(c_in)
            self.one_ch = True
        elif atten_type=='cross_attention':
            self.cmam = CrossAttentinLayer(c_in)
            self.one_ch = False
        elif atten_type=='sel_cross_attention':
            self.cmam = SelectAttentionLayer(c_in)
            self.one_ch = False
        else:
            raise(KeyError, 'No such attention layer named:%s' % atten_type)

        self.detector = Hierarchy(c_in, classes)
        if self.inf_3dnet_weight is not None:
            self.inf_feat_net.load_state_dict(torch.load(self.inf_3dnet_weight))

        self.dw_ratio = 4
        # Don't train feature layers
        for name, para in self.flow_feat_net.named_parameters():
            para.requires_grad = False
        for name, para in self.inf_feat_net.named_parameters():
            para.requires_grad = False

        self.to(self.default_device)
        self.flow_feat_net.to(device_2)
        self.inf_feat_net.to(device_3)
        # self.flow_est.encoder.to(device_2)

    def forward(self, inp):
        # nn.utils.clip_grad_norm_(self.cmam.parameters(), max_norm=)
        # nn.utils.clip_grad_norm_(self.detector.parameters(), max_norm)

        inf = inp[0]
        b,c,t,h,w = inf.shape
        # assert b==1
        #
        # # flow = inf
        # inf_x = inf[:,:,:,::self.dw_ratio,::self.dw_ratio].permute(0,2,1,3,4).squeeze(0)
        # inf_y = inf_x[1:, ...]
        # flow_in = torch.cat([inf_x[0:t-1,...], inf_y], dim=1)
        # assert flow_in.shape[0]==t-1
        #
        # flow = self.flow_est(flow_in)
        # flow = torch.cat([flow, torch.zeros((1, flow.shape[1], h//self.dw_ratio,w//self.dw_ratio)).float().to(self.default_device)], dim=0)
        #
        # inf = inf.to(device_3)
        # inf_f = self.inf_feat_net.extract_features(inf)
        # inf_f = inf_f.to(self.default_device)
        # del inf
        # if self.one_ch:
        #     x, attention = self.cmam(inf_f)
        # else:
        #     flow_p = Variable(flow.to(device_2))
        #     flow_p = F.interpolate(flow_p, scale_factor=self.dw_ratio, mode='bilinear', align_corners=False)
        #     flow_p = flow_p.unsqueeze(0).permute(0, 2, 1, 3, 4)
        #
        #     flow_f = self.flow_feat_net.extract_features(flow_p)
        #     flow_f = flow_f.to(self.default_device)
        #     del flow_p
        #     try:
        #         x, attention = self.cmam(inf_f, flow_f)
        #     except ValueError:
        #         print('WTF')
        # x = [x, inp[1]]
        # x = self.detector(x)
        # return x, flow.permute(1,0,2,3).unsqueeze(0)
        b_data = inf[:, :, :, ::self.dw_ratio, ::self.dw_ratio].permute(0, 2, 1, 3, 4)
        b_flow = []
        for i in range(b):
            inf_x = b_data[i]
            inf_y = inf_x[1:, ...]
            flow_in = torch.cat([inf_x[0:t - 1, ...], inf_y], dim=1)
            assert flow_in.shape[0] == t - 1
            flow = self.flow_est(flow_in)
            flow = torch.cat([flow, torch.zeros((1, flow.shape[1], h // self.dw_ratio, w // self.dw_ratio)).float().to(
                self.default_device)], dim=0)
            b_flow.append(flow)

        inf = inf.to(device_3)
        inf_f = self.inf_feat_net.extract_features(inf)
        inf_f = inf_f.to(self.default_device)
        if self.one_ch:
            x, attention = self.cmam(inf_f)
        else:
            org_flow = torch.zeros(b, flow.shape[1], t, h, w).to(device_2)
            for i, flow in enumerate(b_flow):
                flow_p = Variable(flow)
                flow_p = F.interpolate(flow_p, scale_factor=self.dw_ratio, mode='bilinear', align_corners=False)
                org_flow[i, ...] = flow_p.permute(1, 0, 2, 3)
                # flow_p = flow_p.unsqueeze(0).permute(0, 2, 1, 3, 4)

            flow_f = self.flow_feat_net.extract_features(org_flow)
            flow_f = flow_f.to(self.default_device)
            x, attention = self.cmam(inf_f, flow_f)
        x = [x, inp[1]]
        x = self.detector(x)
        return x, flow.permute(1, 0, 2, 3).unsqueeze(0)


if __name__ == '__main__':
    d_in = torch.rand(1, 3, 100, 224, 224).cuda(3)
    mask = torch.ones(1, 100//8).cuda(3)
    m = InfDetNet('sel_cross_attention',
                  rgb_weight='../../models/infar_i3d.pt000420.pt',
                  classes=3, device=torch.device('cuda:3'))

    tic = time.time()
    for _ in range(100):
        m([d_in, torch.sum(mask, 1)])

    total_time = time.time()-tic
    fps = 1e4/total_time
    print('FPS:%f'%fps)

    from utils.thop import profile, clever_format
    macs, params = profile(m, inputs=([d_in, torch.sum(mask, 1)],))
    macs, params = clever_format([macs, params], "%.3f")
    print('Macs:' + macs + ', Params:' + params)

    # d_in = torch.rand(1, 1024, 100, 1, 1)
    # m = SelectAttentionLayer(1024)
    #
    # from utils.thop import profile, clever_format
    #
    # macs, params = profile(m, inputs=(d_in, d_in,))
    # macs, params = clever_format([macs, params], "%.3f")
    # print('Macs:' + macs + ', Params:' + params)
