import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
W=256
H=128
B=4
D=24
dis = torch.arange(0, D).view(1, -1).repeat(H, 1)
dis=dis.view(1, 1, H, W).repeat(B, 1, 1, 1)

xx = torch.arange(0, W).view(1, -1).repeat(H, 1)#arange产生的是10-1=9 由1-9组成的1维度张量 ，类型int    repeat：复制张量H份    View：改变张量形状   view(1, -1)一维

yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
print(xx)
