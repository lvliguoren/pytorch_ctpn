import cv2
from torchvision import transforms
from model.utils import *
import torch
import numpy as np


mean_vals = [0.471, 0.448, 0.408]
std_vals = [0.234, 0.239, 0.242]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def predict(file_path):
    img = cv2.imread(file_path)
    input = transforms.ToTensor()(img)
    input = transforms.Normalize(mean=mean_vals, std=std_vals)(input)
    input = input.unsqueeze(0) #增加了一个batch维度
    input = input.to(device)

    model = torch.load('model/ctpn.pth').to(device)
    model.eval()
    pre_cls, pre_reg, pre_cls_prob = model(input)
    pre_reg = pre_reg.cpu().numpy()
    pre_cls_prob = pre_cls_prob.cpu().numpy()

    h, w, c = img.shape
    base_anchor = gen_anchor((int(h/16), int(w/16)), 16)
    bbox = bbox_transfor_inv(base_anchor, pre_reg)

    fg_idx = np.where(pre_cls_prob[0,:,1] > 0.9)[0] #筛选出概率大于阈值的行号
    selected_anchor = bbox[fg_idx, :]
    selected_score = pre_cls_prob[0, fg_idx, 1]
    keep_index = filter_bbox(selected_anchor, 16)
    selected_anchor = selected_anchor[keep_index]
    selected_score = selected_score[keep_index]

    #nms
    selected_score = np.reshape(selected_score,(-1, 1))
    nmsbox = np.hstack((selected_anchor, selected_score))
    keep = nms(nmsbox, 0.3)
    selected_anchor = selected_anchor[keep]
    selected_score = selected_score[keep]

    #文本线计算
    textConn = TextProposalConnectorOriented()
    text = textConn.get_text_lines(selected_anchor, selected_score, [h, w])

    for i in text:
        score = str(round(i[8]*100, 2)) + '%'
        cv2.line(img, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 0, 255), 2)
        cv2.line(img, (int(i[0]), int(i[1])), (int(i[4]), int(i[5])), (0, 0, 255), 2)
        cv2.line(img, (int(i[6]), int(i[7])), (int(i[2]), int(i[3])), (0, 0, 255), 2)
        cv2.line(img, (int(i[6]), int(i[7])), (int(i[4]), int(i[5])), (0, 0, 255), 2)
        cv2.putText(img, score, (int(i[0])+13, int(i[1])+13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,0),
                    2,
                    cv2.LINE_AA)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    predict('C:/Users/Administrator/Desktop/2.jpg')