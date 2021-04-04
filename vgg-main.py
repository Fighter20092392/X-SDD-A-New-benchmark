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

class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        # 定义卷积层和池化层，共13层卷积，5层池化
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 简化版全连接层
 #       self.classifier = nn.Sequential(
 #           nn.Linear(4 * 4 * 512, 1024),
 #           nn.ReLU(),
 #           nn.Dropout(p=0.5),
#            nn.Linear(1024, 1024),
#            nn.ReLU(),
#            nn.Dropout(p=0.5),
#            nn.Linear(1024, 2)
#        )

        #VGG-16的全连接层
        self.classes = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 7)
        )

    # 定义每次执行的计算步骤
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 7*7*512)
        x = self.classes(x)
        return x

model=VGG16()
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
    torch.save(model, r'E:\vgg16_save.pth')

if __name__ == '__main__':
    train()
