import numpy as np


def gen_anchor(featuresize, scale):
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)

    base_anchor = np.array([0, 0, 15, 15])
    # center x,y
    xt = (base_anchor[0] + base_anchor[2]) * 0.5
    yt = (base_anchor[1] + base_anchor[3]) * 0.5

    # x1 y1 x2 y2
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5
    base_anchor = np.hstack((x1, y1, x2, y2))

    h, w = featuresize
    shift_x = np.arange(0, w) * scale
    shift_y = np.arange(0, h) * scale
    # apply shift
    anchor = []
    for i in shift_y:
        for j in shift_x:
            anchor.append(base_anchor + [j, i, j, i])
    return np.array(anchor).reshape((-1, 4))


def cal_iou(box1, box1_area, boxes2, boxes2_area):
    x1 = np.maximum(box1[0], boxes2[:,0])
    y1 = np.maximum(box1[1], boxes2[:,1])
    x2 = np.minimum(box1[2], boxes2[:,2])
    y2 = np.minimum(box1[3], boxes2[:,3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    iou = intersection / (box1_area + boxes2_area[:] - intersection[:])
    return iou


def cal_overlaps(boxes1, boxes2):
    """
    boxes1 [x1,y1,x2,y2]  anchor
    boxes2 [x1,y1,x2,y2]  grouth-box
    """
    area1 = (boxes1[:, 0] - boxes1[:, 2]) * (boxes1[:, 1] - boxes1[:, 3])
    area2 = (boxes2[:, 0] - boxes2[:, 2]) * (boxes2[:, 1] - boxes2[:, 3])

    overlaps = np.zeros(shape=(boxes1.shape[0], boxes2.shape[0]))

    for i in range(boxes1.shape[0]):
        overlaps[i][:] = cal_iou(boxes1[i], area1[i], boxes2, area2)

    return overlaps


def bbox_transfrom(anchors, gtboxes):
    """
     compute relative predicted vertical coordinates Vc ,Vh
        with respect to the bounding box location of an anchor
    """
    Cy = (gtboxes[:, 1] + gtboxes[:, 3]) * 0.5
    Cya = (anchors[:, 1] + anchors[:, 3]) * 0.5
    h = gtboxes[:, 3] - gtboxes[:, 1] + 1.0
    ha = anchors[:, 3] - anchors[:, 1] + 1.0

    Vc = (Cy - Cya) / ha
    Vh = np.log(h / ha)

    return np.vstack((Vc, Vh)).transpose()


def cal_rpn(imgsize, featuresize, scale, gtboxes):
    imgh, imgw = imgsize

    # gen base anchor
    base_anchor = gen_anchor(featuresize, scale)

    # calculate iou len(anchor)*len(gtboxes)
    overlaps = cal_overlaps(base_anchor, gtboxes)

    # init labels -1 don't care  0 is negative  1 is positive
    labels = np.empty(base_anchor.shape[0])
    labels.fill(-1)

    for i in range(labels.shape[0]):
        # IOU > IOU_POSITIVE
        if np.max(overlaps[i] > 0.7):
            labels[i] = 1
        # IOU <IOU_NEGATIVE
        if np.max(overlaps[i] < 0.3):
            labels[i] = 0

    # ensure that every GT box has at least one positive RPN region
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    labels[gt_argmax_overlaps] = 1

    # only keep anchors inside the image
    outside_anchor = np.where(
        (base_anchor[:, 0] < 0)|
        (base_anchor[:, 1] < 0)|
        (base_anchor[:, 2] >= imgw)|
        (base_anchor[:, 3] >= imgh)
    )[0]
    labels[outside_anchor] = -1

    # subsample positive labels ,if greater than RPN_POSITIVE_NUM(default 128)
    fg_index = np.where(labels == 1)[0]
    if len(fg_index) > 150 :
        labels[np.random.choice(fg_index, len(fg_index) - 150, replace=False)] = -1

    # subsample negative labels
    bg_index = np.where(labels ==  0)[0]
    num_bg = 300 - np.sum(labels == 1)
    if len(bg_index) > num_bg:
        labels[np.random.choice(bg_index, len(bg_index) - num_bg, replace=False)] = -1

    # calculate bbox targets
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    # 每个anchor和它对应的iou最大的gt 计算偏移量
    bbox_targets = bbox_transfrom(base_anchor, gtboxes[anchor_argmax_overlaps, :])

    return labels, bbox_targets, base_anchor