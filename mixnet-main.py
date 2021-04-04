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

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        return x * self.sigmoid(x)
 
 
NON_LINEARITY = {
    'ReLU': nn.ReLU(inplace=True),
    'Swish': Swish(),
}
 
 
def _RoundChannels(c, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    return new_c
 
 
def _SplitChannels(channels, num_groups):
    split_channels = [channels // num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels
 
 
def Conv3x3Bn(in_channels, out_channels, stride, non_linear='ReLU'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        NON_LINEARITY[non_linear]
    )
 
 
def Conv1x1Bn(in_channels, out_channels, non_linear='ReLU'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        NON_LINEARITY[non_linear]
    )
 
 
class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, se_ratio):
        super(SqueezeAndExcite, self).__init__()
 
        squeeze_channels = channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')
 
        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = NON_LINEARITY['Swish']
        self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()
 
    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y
 
        return y
 
 
class MDConv(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(MDConv, self).__init__()
 
        self.num_groups = len(kernel_size)
        self.split_channels = _SplitChannels(channels, self.num_groups)
 
        self.mixed_depthwise_conv = nn.ModuleList([])
        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(nn.Conv2d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i] // 2,
                groups=self.split_channels[i],
                bias=False
            ))
 
    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)
 
        x_split = torch.split(x, self.split_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)
 
        return x
 
 
class MixNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, non_linear='ReLU', se_ratio=0.0):
        super(MixNetBlock, self).__init__()
 
        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        se = (se_ratio != 0.0)
        self.residual_connection = (stride == 1 and in_channels == out_channels)
 
        conv = []
 
        if expand:
            # expansion phase
            pw_expansion = nn.Sequential(
                nn.Conv2d(in_channels, expand_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(expand_channels),
                NON_LINEARITY[non_linear]
            )
            conv.append(pw_expansion)
 
        # depthwise convolution phase
        dw = nn.Sequential(
            MDConv(expand_channels, kernel_size, stride),
            nn.BatchNorm2d(expand_channels),
            NON_LINEARITY[non_linear]
        )
        conv.append(dw)
 
        if se:
            # squeeze and excite
            squeeze_excite = nn.Sequential(SqueezeAndExcite(expand_channels, se_ratio))
            conv.append(squeeze_excite)
 
        # projection phase
        pw_projection = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        conv.append(pw_projection)
 
        self.conv = nn.Sequential(*conv)
 
    def forward(self, x):
        if self.residual_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)
 
 
class MixNet(nn.Module):
    # [in_channels, out_channels, kernel_size, stride, expand_ratio, non_linear, se_ratio]
    mixnet_s = [(16, 16, [3], 1, 1, 'ReLU', 0.0),
                (16, 24, [3], 2, 6, 'ReLU', 0.0),
                (24, 24, [3], 1, 3, 'ReLU', 0.0),
                (24, 40, [3, 5, 7], 2, 6, 'Swish', 0.5),
                (40, 40, [3, 5], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], 1, 6, 'Swish', 0.5),
                (40, 80, [3, 5, 7], 2, 6, 'Swish', 0.25),
                (80, 80, [3, 5], 1, 6, 'Swish', 0.25),
                (80, 80, [3, 5], 1, 6, 'Swish', 0.25),
                (80, 120, [3, 5, 7], 1, 6, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5),
                (120, 200, [3, 5, 7, 9, 11], 2, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5)]
 
    mixnet_m = [(24, 24, [3], 1, 1, 'ReLU', 0.0),
                (24, 32, [3, 5, 7], 2, 6, 'ReLU', 0.0),
                (32, 32, [3], 1, 3, 'ReLU', 0.0),
                (32, 40, [3, 5, 7, 9], 2, 6, 'Swish', 0.5),
                (40, 40, [3, 5], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], 1, 6, 'Swish', 0.5),
                (40, 40, [3, 5], 1, 6, 'Swish', 0.5),
                (40, 80, [3, 5, 7], 2, 6, 'Swish', 0.25),
                (80, 80, [3, 5, 7, 9], 1, 6, 'Swish', 0.25),
                (80, 80, [3, 5, 7, 9], 1, 6, 'Swish', 0.25),
                (80, 80, [3, 5, 7, 9], 1, 6, 'Swish', 0.25),
                (80, 120, [3], 1, 6, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5),
                (120, 120, [3, 5, 7, 9], 1, 3, 'Swish', 0.5),
                (120, 200, [3, 5, 7, 9], 2, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5),
                (200, 200, [3, 5, 7, 9], 1, 6, 'Swish', 0.5)]
 
    def __init__(self, net_type='mixnet_s', input_size=512, num_classes=7
                 , stem_channels=16, feature_size=1536,
                 depth_multiplier=1.0):
        super(MixNet, self).__init__()
 
        if net_type == 'mixnet_s':
            config = self.mixnet_s
            stem_channels = 16
            dropout_rate = 0.2
        elif net_type == 'mixnet_m':
            config = self.mixnet_m
            stem_channels = 24
            dropout_rate = 0.25
        elif net_type == 'mixnet_l':
            config = self.mixnet_m
            stem_channels = 24
            depth_multiplier *= 1.3
            dropout_rate = 0.25
        else:
            raise TypeError('Unsupported MixNet type')
 
        assert input_size % 32 == 0
 
        # depth multiplier
        if depth_multiplier != 1.0:
            stem_channels = _RoundChannels(stem_channels * depth_multiplier)
 
            for i, conf in enumerate(config):
                conf_ls = list(conf)
                conf_ls[0] = _RoundChannels(conf_ls[0] * depth_multiplier)
                conf_ls[1] = _RoundChannels(conf_ls[1] * depth_multiplier)
                config[i] = tuple(conf_ls)
 
        # stem convolution
        self.stem_conv = Conv3x3Bn(3, stem_channels, 2)
 
        # building MixNet blocks
        layers = []
        for in_channels, out_channels, kernel_size, stride, expand_ratio, non_linear, se_ratio in config:
            layers.append(
                MixNetBlock(in_channels, out_channels, kernel_size, stride, expand_ratio, non_linear, se_ratio))
        self.layers = nn.Sequential(*layers)
 
        # last several layers
        self.head_conv = Conv1x1Bn(config[-1][1], feature_size)
 
        self.avgpool = nn.AvgPool2d(input_size // 32, stride=1)
        self.classifier = nn.Linear(feature_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
 
        self._initialize_weights()
 
    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.head_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.dropout(x)
 
        return x
 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

model = MixNet(input_size=224)
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
    torch.save(model, '/home/fighter/Downloads/mixnet_save.pth')


if __name__ == '__main__':
    train()
