import torch
import torch.nn as nn
import torchvision.models as models
import os


class BiLSTM(nn.Module):
    def __init__(self, in_planes, hidden_unit_num, out_planes):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(in_planes, hidden_unit_num, bidirectional=True)
        self.fc = nn.Linear(hidden_unit_num*2, out_planes)

    def forward(self, x):
        out, _ = self.lstm(x)
        NH, W, C = out.size()
        out = out.view(NH*W, C)
        out = self.fc(out)
        out = out.view(NH, W, -1)

        return out


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
        self.rpn = nn.Conv2d(512, 512, 3, 1, 1)
        self.brnn = BiLSTM(512, 128, 512)
        # self.fc = BasicConv(256, 512, 1, 1)
        self.rpn_class = nn.Linear(512, 10*2)
        self.rpn_regress = nn.Linear(512, 10*2)
        self.rpn_class_softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.base_layers(x)
        x = self.rpn(x)
        N, C, H, W = x.size()

        x = x.permute(0,2,3,1).contiguous()  # channels last
        x = x.view(N*H, W, C)
        x = self.brnn(x)

        x = x.view(N*H*W, 512)
        cls = self.rpn_class(x)
        # clsH*cls_w*10个anchor与target保持一致
        cls = cls.view(N, H*W*10, 2)
        # cls = self.rpn_class_softmax(cls)
        # cls = cls.permute(cls_N, 20, cls_H, cls_W).contiguous()

        cls_prob = self.rpn_class_softmax(cls)

        reg = self.rpn_regress(x)
        reg = reg.view(N, H*W*10, 2)

        return cls, reg, cls_prob


class RPN_ClS_LOSS(nn.Module):
    def __init__(self, device):
        super(RPN_ClS_LOSS, self).__init__()
        self.critetion = nn.CrossEntropyLoss().to(device)

    def forward(self, pred, target) :
        # pred N*(H*W*10)*2
        # target N*(H*W*10)
        target_idx = (target[0]!= -1).nonzero()[:,0] #有效数据，不为-1的target_idx
        # s = pred[0][target_idx]
        # m = F.softmax(s,dim=-1)
        loss = self.critetion(pred[0][target_idx], target[0][target_idx].long())
        return loss


class RPN_REG_LOSS(nn.Module):
    def __init__(self, device):
        super(RPN_REG_LOSS, self).__init__()
        self.device = device
        self.critetion = nn.SmoothL1Loss().to(device)

    def forward(self, pred, target):
        # pred N*(H*W*10)*2
        # target N*(H*W*10)*3(cls,Vc,Vh)
        target_cls = target[0][:,0]
        target_reg = target[0][:,1:3]
        target_idx = (target_cls == 1).nonzero()[:, 0]  #只学习正样本
        if target_idx.numel() > 0:
            loss = self.critetion(pred[0][target_idx], target_reg[target_idx])
        else:
            loss = torch.tensor(0.0).to(self.device)

        return loss





