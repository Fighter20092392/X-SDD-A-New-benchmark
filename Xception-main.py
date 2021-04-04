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
import torch
import torch.nn as nn

def ConvBN(in_channels,out_channels,kernel_size,stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=0 if kernel_size==1 else (kernel_size-1)//2),
        nn.BatchNorm2d(out_channels),
    )

def ConvBNRelu(in_channels,out_channels,kernel_size,stride):
    return nn.Sequential(
        ConvBN(in_channels, out_channels, kernel_size, stride),
        nn.ReLU6(inplace=False),
    )

def SeparableConvolution(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,padding=1,groups=in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
    )
def SeparableConvolutionRelu(in_channels, out_channels):
    return nn.Sequential(
        SeparableConvolution(in_channels, out_channels),
        nn.ReLU6(inplace=False),
    )

def ReluSeparableConvolution(in_channels, out_channels):
    return nn.Sequential(
        nn.ReLU6(inplace=False),
        SeparableConvolution(in_channels, out_channels)
    )

class EntryBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, first_relu=True):
        super(EntryBottleneck, self).__init__()
        mid_channels = out_channels

        self.shortcut = ConvBN(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=2)

        self.bottleneck = nn.Sequential(
            ReluSeparableConvolution(in_channels=in_channels,out_channels=mid_channels) if first_relu else SeparableConvolution(in_channels=in_channels,out_channels=mid_channels),
            ReluSeparableConvolution(in_channels=mid_channels, out_channels=out_channels),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

    def forward(self, x):
        out = self.shortcut(x)
        x = self.bottleneck(x)
        return out+x


class MiddleBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiddleBottleneck, self).__init__()
        mid_channels = out_channels

        self.bottleneck = nn.Sequential(
            ReluSeparableConvolution(in_channels=in_channels,out_channels=mid_channels),
            ReluSeparableConvolution(in_channels=mid_channels, out_channels=mid_channels),
            ReluSeparableConvolution(in_channels=mid_channels, out_channels=out_channels),
        )

    def forward(self, x):
        out = self.bottleneck(x)
        return out+x

class ExitBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExitBottleneck, self).__init__()
        mid_channels = in_channels

        self.shortcut = ConvBN(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=2)

        self.bottleneck = nn.Sequential(
            ReluSeparableConvolution(in_channels=in_channels,out_channels=mid_channels),
            ReluSeparableConvolution(in_channels=mid_channels, out_channels=out_channels),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

    def forward(self, x):
        out = self.shortcut(x)
        x = self.bottleneck(x)
        return out+x

class Xception(nn.Module):
    def __init__(self, num_classes=7):
        super(Xception, self).__init__()

        self.entryFlow = nn.Sequential(
            ConvBNRelu(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            ConvBNRelu(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            EntryBottleneck(in_channels=64, out_channels=128, first_relu=False),
            EntryBottleneck(in_channels=128, out_channels=256, first_relu=True),
            EntryBottleneck(in_channels=256, out_channels=728, first_relu=True),
        )
        self.middleFlow = nn.Sequential(
            MiddleBottleneck(in_channels=728,out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
            MiddleBottleneck(in_channels=728, out_channels=728),
        )
        self.exitFlow = nn.Sequential(
            ExitBottleneck(in_channels=728, out_channels=1024),
            SeparableConvolutionRelu(in_channels=1024, out_channels=1536),
            SeparableConvolutionRelu(in_channels=1536, out_channels=2048),
            nn.AdaptiveAvgPool2d((1,1)),
        )

        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.entryFlow(x)
        x = self.middleFlow(x)
        x = self.exitFlow(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out

model=Xception()
# 定义超级参数
epoch_n = 100

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
    torch.save(model, '/home/fighter/Downloads/Xception_save.pth')



if __name__ == '__main__':
    train()
