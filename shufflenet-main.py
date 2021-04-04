import torch
import time
import os
import random
import math
import torchvision
import torch.nn as nn
from data.dataset import *
from tqdm import tqdm
import torch.nn.functional as F
from config import Config
import matplotlib.pyplot as plt
import torch.nn.init as init

# 数据类实例
dst = Dataset()
# 配置类实例
conf = Config()
# 模型类实例
class Bottleneck(nn.Module):  # shuffle Net 模仿的是resnet bottleblock的结构
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_planes = out_planes // 4  # 每个ShuffleNet unit的bottleneck通道数为输出的1/4(和ResNet设置一致)
        self.groups = 1 if in_planes == 24 else groups  # 第一层卷积之后是24，所以不必group
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=self.groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)  # 这里应该用dw conv的
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))  # 每个阶段第一个block步长是2，下个阶段通道翻倍

    @staticmethod
    def shuffle_channels(x, groups):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,W] -> [N,C,H,W]'''
        '''一共C个channel要分成g组混合的channel，先把C reshape成(g, C/g)的形状，然后转置成(C/g, g)最后平坦成C组channel'''
        N, C, H, W = x.size()
        return x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)  # 因为x之前view过了，他的内存不连续了，需要contiguous来规整一下

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle_channels(out, self.groups)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.stride == 2:
            res = self.shortcut(x)
            out = F.relu(torch.cat([out, res], 1))  # shuffle-net 对做了下采样的res用的是cat而不是+
        else:
            out = F.relu(out + x)
        return out


class ShuffleNet(nn.Module):

    def __init__(self, out_planes, num_blocks, groups, num_classes=7, depth_multiplier=1.):  # depth_multiplier是控制通道数的缩放因子
        super(ShuffleNet, self).__init__()
        self.num_classes = num_classes
        self.in_planes = int(24 * depth_multiplier)  # 通常第一个卷积的输出为24
        self.out_planes = [int(depth_multiplier * x) for x in out_planes]

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.bn1 = nn.BatchNorm2d(24)
        # self.in_planes = 24
        layers = []
        for out_plane, num_block in zip(out_planes, num_blocks):
            layers.append(self._make_layer(out_plane, num_block, groups))
        self.layers = nn.ModuleList(layers)  # 把list里面的每一个元素变成一个module
        if num_classes is not None:
            self.avgpool = nn.AdaptiveMaxPool2d(1)
            self.fc = nn.Linear(out_planes[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1  # 如果是第一个block就是2 否则就是1
            cat_planes = self.in_planes if i == 0 else 0  # 因为第一个要下采样并且cat，所以为了下一个block加的时候能够通道匹配，要先减掉cat的通道数
            layers.append(Bottleneck(self.in_planes, out_planes - cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes  # 第一个过后input就都是out_planes
        return nn.Sequential(*layers)

    @property
    def layer_channels(self):
        return self.out_planes

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        c = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            c.append(x)

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        else:  # 返回每一个阶段的特征
            return c


def shufflenet(**kwargs):  # group = 3 论文实验中效果最好的
    planes = [240, 480, 960]
    layers = [4, 8, 4]
    model = ShuffleNet(planes, layers, groups=3, **kwargs)
    return model


def shufflenet_4(**kwargs):  # group = 4
    planes = [272, 544, 1088]
    layers = [4, 8, 4]
    model = ShuffleNet(planes, layers, groups=4, **kwargs)
    return model


def shufflenet_2(**kwargs):
    planes = [200, 400, 800]
    layers = [4, 8, 4]
    model = ShuffleNet(planes, layers, groups=2, **kwargs)
    return model


def shufflenet_1(**kwargs):
    planes = [144, 288, 576]
    layers = [4, 8, 4]
    model = ShuffleNet(planes, layers, groups=1, **kwargs)
    return model

model= shufflenet()
# 定义超级参数
epoch_n = 50

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 如果GPUs可用，则将模型上需要计算的所有参数复制到GPUs上
if torch.cuda.is_available():
    model = model.cuda()

def train():
    for epoch in range(1, epoch_n + 1):
        print('Epoch {}/{}'.format(epoch, epoch_n))
        print('-'*20)

        for phase in ['train', 'valid']:
            if phase == 'train':
                print('Training...')
                # 打开训练模式
                model.train(True)
            else:
                print('Validing...')
                # 关闭训练模式
                model.train(False)

            # 损失值
            running_loss = 0.0
            # 预测的正确数
            running_correct = 0
            # 让batch的值从1开始，便于后面计算
            for batch, data in enumerate(dst.data_images_loader[phase], 1):
                # 实际输入值和输出值
                X, y = data
                # 将参数复制到GPUs上进行运算
                if torch.cuda.is_available():
                    X, y = X.cuda(), y.cuda()
                # outputs.shape = [32,2] -> [1,2]
                outputs = model(X)
                # 从输出结果中取出需要的预测值
                _, y_pred = torch.max(outputs.detach(),  1)
                # 将Varibale的梯度置零
                optimizer.zero_grad()
                # 计算损失值
                loss = loss_fn(outputs, y)
                if phase == 'train':
                    # 反向传播求导
                    loss.backward()
                    # 更新所有参数
                    optimizer.step()

                running_loss += loss.detach().item()
                running_correct += torch.sum(y_pred == y)
                if batch % 500 == 0 and phase == 'train':
                    print('Batch {}/{},Train Loss:{:.2f},Train Acc:{:.2f}%'.format(
                        batch, len(dst.data_images[phase])/conf.batch_size, running_loss/batch, 100*running_correct.item()/(conf.batch_size*batch)
                    ))
            epoch_loss = running_loss*conf.batch_size/len(dst.data_images[phase])
            epoch_acc = 100*running_correct.item()/len(dst.data_images[phase])
            print('{} Loss:{:.2f} Acc:{:.2f}%'.format(phase, epoch_loss, epoch_acc))
    print('Saving state')
    torch.save(model, '/home/fighter/Downloads/shufflenet_save.pth')




if __name__ == '__main__':
    train()
