# Author: Guoqing Bao
# School of Computer Science, The University of Sydney
# Date: 2019-12-12
# GitHub Project Link: https://github.com/guoqingbao/Pathofusion
# Please cite our work if you found it is useful for your research or clinical practice

import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class SubNet(nn.Module):

    def __init__(self, n_out, init_conv_batch=True):
        super(SubNet, self).__init__()

        self.conv = nn.Conv2d(3, 16, 3, padding=1, stride=2)
        self.block1 = self.make_block(4, 16, 64, 2)
        self.block2 = self.make_block(4, 64, 128, 2)
        self.block3 = self.make_block(4, 128, 256, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(256, n_out)  
        
        # Initialize paramters
        for m in self.modules():
            if init_conv_batch:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(x)
        out = self.pool(x)
        out = out.view(out.size(0), -1)
        return self.linear(out)

    def make_block(self, n_pairs, in_channel, out_channel, stride):
        layers = []
        # in total of 8 consecutive convolution/batch norm in a block
        for i in range(int(n_pairs)):
            layers.append(
                nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(out_channel, out_channel, 3, padding=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_channel)
                    )
            )
            in_channel, stride = out_channel, 1 # only first convolution have a stride of 2
        return nn.Sequential(*layers)

# The BDCNN composed of two subnets
class BDCNN(nn.Module):
    def __init__(self, n_classes, init_conv_batch=True):
        super(BDCNN, self).__init__()
        self.path1 = SubNet(16, init_conv_batch)
        self.path2 = SubNet(16, init_conv_batch)

        self.classifier = nn.Linear(32, n_classes)

                
    def forward(self, x1, x2):
        path1 = self.path1(x1)
        path2 = self.path2(x2)
        x = torch.cat((path1, path2), dim=1)
        x = self.classifier(x)
        return x