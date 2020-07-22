import numpy as np
import torch


# a = [0,1,1,-1,1,0,1,-1,-1,0]
# print(a)
# b = np.expand_dims(a, axis=0)
# print(b)
# c = torch.from_numpy(b).float()
# y_true = c[0]
# print(y_true)
# print(y_true!=-1)
# cls_keep = (y_true != -1).nonzero()
# print(cls_keep)
# d = cls_keep[:, 0]
# print(d)
# cls_true = y_true[cls_keep].long()
# print(cls_true)


# a = np.random.random(size=(5,6))
# print(a)
# print("------------------")
# b = [0,1,2]
# c= a[b]
# print(c)
# print("------------------")
# d= a[b][1:2]
# print(d)
#
# torch.nn.SmoothL1Loss
h,w,c = 500,600,3
k = np.array([h, w, c])
print(k)
m = k.reshape([1, 3])
print(m)


