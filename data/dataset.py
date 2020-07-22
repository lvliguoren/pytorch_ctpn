from torch.utils.data import Dataset
import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from model.utils import cal_rpn
from torchvision import transforms
import torch


mean_vals = [0.471, 0.448, 0.408]
std_vals = [0.234, 0.239, 0.242]


def readxml(path):
    gtboxes = []
    imgfile = ''
    xml = ET.parse(path)
    for elem in xml.iter():
        if 'filename' in elem.tag:
            imgfile = elem.text
        if 'object' in elem.tag:
            for attr in list(elem):
                if 'bndbox' in attr.tag:
                    xmin = int(round(float(attr.find('xmin').text)))
                    ymin = int(round(float(attr.find('ymin').text)))
                    xmax = int(round(float(attr.find('xmax').text)))
                    ymax = int(round(float(attr.find('ymax').text)))

                    gtboxes.append((xmin, ymin, xmax, ymax))

    return np.array(gtboxes), imgfile


class VOCDataset(Dataset):
    def __init__(self, datadir, labelsdir):
        super(Dataset,self).__init__()
        '''
        :param txtfile: image name list text file
        :param datadir: image's directory
        :param labelsdir: annotations' directory
        '''
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))
        if not os.path.isdir(labelsdir):
            raise Exception('[ERROR] {} is not a directory'.format(labelsdir))

        self.datadir = datadir
        self.img_names = os.listdir(self.datadir)
        self.labelsdir = labelsdir

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.datadir, img_name)
        xml_path = os.path.join(self.labelsdir, img_name.replace('.jpg', '.xml'))
        gtbox, _ = readxml(xml_path)
        img = cv2.imread(img_path)
        h, w, c = img.shape

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean= mean_vals, std=std_vals)(img)
        cls, reg, base_anchor = cal_rpn((h,w), (int(h/16),int(w/16)), 16, gtbox)

        reg = np.hstack((cls.reshape(-1,1),reg))
        cls = torch.from_numpy(cls).float()
        reg = torch.from_numpy(reg).float()

        return img, cls, reg


if __name__ == '__main__':
    img_path = 'E:/TEST/VOC2007/JPEGImages/img_1002.jpg'
    xml_path = 'E:/TEST/VOC2007/Annotations/img_1002.xml'
    # img_path = 'C:/Users/Administrator/Desktop/VOCdevkit(1)/VOCdevkit/VOC2007/JPEGImages/img_1002.jpg'
    # xml_path = 'C:/Users/Administrator/Desktop/VOCdevkit(1)/VOCdevkit/VOC2007/Annotations/img_1002.xml'
    gtbox, _ = readxml(xml_path)
    img = cv2.imread(img_path)

    h, w, c = img.shape
    cls, reg, base_anchor = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)
    assert len(cls) == len(base_anchor)
    cls_keep = (cls==1).nonzero()
    cls_anchor = base_anchor[cls_keep]

    for anchor in cls_anchor:
        x1,y1,x2,y2 = int(anchor[0]),int(anchor[1]),int(anchor[2]),int(anchor[3])
        cv2.line(img,(x1,y2),(x2,y2),(0,0,255),2)
        cv2.line(img,(x1,y1),(x2,y1),(0,0,255),2)
        cv2.line(img,(x1,y2),(x1,y1),(0,0,255),2)
        cv2.line(img,(x2,y2),(x2,y1),(0,0,255),2)


    # for gt in gtbox:
    #     x1,y1,x2,y2 = gt[0],gt[1],gt[2],gt[3]
    #     cv2.line(img,(x1,y2),(x2,y2),(0,0,255),2)
    #     cv2.line(img,(x1,y1),(x2,y1),(0,0,255),2)
    #     cv2.line(img,(x1,y2),(x1,y1),(0,0,255),2)
    #     cv2.line(img,(x2,y2),(x2,y1),(0,0,255),2)
        # break
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
