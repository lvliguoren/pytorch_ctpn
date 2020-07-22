import torch
import torch.nn as nn
import torchvision.models as models
import os
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 relu=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        torch.nn.init.normal_(self.conv.weight,0.0,0.02)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CTPN(nn.Module):
    def __init__(self, pretrained, pretrained_model_path):
        super(CTPN, self).__init__()

        if pretrained:
            if os.path.exists(pretrained_model_path):
                base_model = models.vgg16(pretrained=False)
                base_model.load_state_dict(torch.load(pretrained_model_path))
            else:
                base_model = models.vgg16(pretrained=True)
        else:
            base_model = models.vgg16()

        layers = list(base_model.features)[:-1]
        self.base_layers = nn.Sequential(*layers)  # block5_conv3 output
        self.rpn = BasicConv(512, 512, 3, 1, 1, relu=False)
        self.brnn = nn.LSTM(512, 128, bidirectional=True)
        self.fc = BasicConv(256, 512, 1, 1)
        self.rpn_class =BasicConv(512, 10*2, 1, 1, relu=False)
        self.rpn_regress = BasicConv(512, 10*2, 1, 1, relu=False)
        self.rpn_class_softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.base_layers(x)
        x = self.rpn(x)
        N, C, H, W = x.size()

        x = x.permute(0,2,3,1).contiguous()  # channels last
        x = x.view(N*H, W, C)
        x, _ = self.brnn(x)

        x = x.view(N, H, W, 256)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.fc(x)

        cls = self.rpn_class(x)
        cls_N, cls_C, cls_H, cls_W = cls.size()
        # clsH*cls_w*10个anchor与target保持一致
        cls = cls.view(cls_N, cls_H*cls_W*10, 2)
        # cls = self.rpn_class_softmax(cls)
        # cls = cls.permute(cls_N, 20, cls_H, cls_W).contiguous()

        cls_prob = self.rpn_class_softmax(cls)

        reg = self.rpn_regress(x)
        reg_N, reg_C, reg_H, reg_W = reg.size()
        reg = reg.view(reg_N, reg_H*reg_W*10, 2)

        return cls, reg, cls_prob


class RPN_ClS_LOSS(nn.Module):
    def __init__(self, device):
        super(RPN_ClS_LOSS, self).__init__()
        self.device = device

    def forward(self, pred, target) :
        # pred N*(H*W*10)*2
        # target N*(H*W*10)
        critetion = nn.CrossEntropyLoss().to(self.device)
        target_idx = (target[0]!= -1).nonzero()[:,0] #有效数据，不为-1的target_idx
        # s = pred[0][target_idx]
        # m = F.softmax(s,dim=-1)
        loss = critetion(pred[0][target_idx], target[0][target_idx].long())
        return loss


class RPN_REG_LOSS(nn.Module):
    def __init__(self, device):
        super(RPN_REG_LOSS, self).__init__()
        self.device = device


    def forward(self, pred, target):
        # pred N*(H*W*10)*2
        # target N*(H*W*10)*3(cls,Vc,Vh)
        critetion = nn.SmoothL1Loss().to(self.device)
        cls = target[0][:,0]
        target = target[0][:,1:3]
        target_idx = (cls == 1).nonzero()[:, 0]  #只学习正样本
        loss = critetion(pred[0][target_idx], target[target_idx])

        return loss





