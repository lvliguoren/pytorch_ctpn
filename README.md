本文是以pytorch实现CTPN的练习之作，主要是以@opconty的[项目](https://github.com/opconty/pytorch_ctpn)为版本，
在模型上做了些修改。之所以修改的原因在于以原文代码训练VOC2007数据集时，出现Loss几乎不会下降的情况，并且Loss达到了0.68，
与原文的0.26相差甚远。我一度以为是数据集出现了问题，后来发现把rpn_class和rpn_regress这两层改为全连接层，模型Loss下降更平滑和快速。
# Loss
Vgg16参数预加载之后，以VOC2007数据集训练，在epoch-5的时候Loss已经下降到0.077
左右。以下是epoch-8的结果：    
```
epoch:8, batch_idx:4523, loss_cls:0.0145, loss_reg:0.0026, loss:0.0171, avg_loss:0.0664
epoch:8, batch_idx:4524, loss_cls:0.1091, loss_reg:0.0026, loss:0.1116, avg_loss:0.0665
epoch:8, batch_idx:4525, loss_cls:0.0339, loss_reg:0.0055, loss:0.0394, avg_loss:0.0665
epoch:8, batch_idx:4526, loss_cls:0.0251, loss_reg:0.0033, loss:0.0284, avg_loss:0.0664
```
# Result
![result](result.jpg)
# Reference
[https://github.com/opconty/pytorch_ctpn](https://github.com/opconty/pytorch_ctpn)   
[https://github.com/eragonruan/text-detection-ctpn](https://github.com/eragonruan/text-detection-ctpn)