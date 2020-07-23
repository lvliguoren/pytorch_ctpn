import torch
import data.dataset as dataset
import model.ctpn as ctpn
from torch.utils.data import DataLoader
import torchvision
import os


base_dir = 'E:/TEST'
img_dir = os.path.join(base_dir, 'VOC2007/JPEGImages')
xml_dir = os.path.join(base_dir, 'VOC2007/Annotations')

train_txt_file = os.path.join(base_dir, r'VOC2007/ImageSets/Main/train.txt')
val_txt_file = os.path.join(base_dir, r'VOC2007/ImageSets/Main/val.txt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ctpn.CTPN(pretrained=True,pretrained_model_path='model/vgg16-397923af.pth').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
critetion_cls = ctpn.RPN_ClS_LOSS(device)
critetion_reg = ctpn.RPN_REG_LOSS(device)

data_set = dataset.VOCDataset(datadir=img_dir, labelsdir=xml_dir)
dataloader =DataLoader(dataset=data_set,batch_size=1,shuffle=True)


def train():
    best_loss_cls = 100
    best_loss_reg = 100
    best_loss = 100

    # pretrained_dict = torch.load('model/vgg16-397923af.pth')
    # model_dict = model.base_layers.state_dict()
    # pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.base_layers.load_state_dict(model_dict)
    # torchvision.models.detection.fasterrcnn_resnet50_fpn()
    # model.apply(weights_init)
    model.train()
    for epoch in range(20):
        epoch_loss_cls = 0
        epoch_loss_regr = 0
        epoch_loss = 0
        epoch_batch = 0
        for batch_idx, (imgs, clss, regs) in enumerate(dataloader):
            imgs = imgs.to(device)
            clss = clss.to(device)
            regs = regs.to(device)

            pre_cls, pre_reg, pre_cls_prob = model(imgs)
            loss_cls = critetion_cls(pre_cls, clss)
            loss_reg = critetion_reg(pre_reg, regs)
            loss = loss_cls + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_cls += loss_cls.item()
            epoch_loss_regr += loss_cls.item()
            epoch_loss += loss.item()
            epoch_batch += 1

            print("epoch:{}, batch_idx:{}, loss_cls:{:.4f}, loss_reg:{:.4f}, loss:{:.4f}, avg_loss:{:.4f}"
                  .format(epoch, batch_idx, loss_cls, loss_reg, loss, epoch_loss/epoch_batch))

            if best_loss_cls > loss_cls or best_loss_reg > loss_reg or best_loss > loss:
                best_loss_cls = loss_cls
                best_loss_reg = loss_reg
                best_loss = loss
                torch.save(model, 'model/ctpn.pth')

    print("--------------END----------------")


if __name__ == '__main__':
    train()
    # print(model)
