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
model = Xception()
model=torch.load('/home/fighter/Downloads/Xception_save.pth')
seed=17
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
def test():
    dst = Dataset(train=False)
    data_loader_test = torch.utils.data.DataLoader(dst,
                                                   batch_size=conf.batch_size,shuffle=False)                                     
    # 保存测试结果
    global resultsx3
    resultsx3= []
    # tqdm模块用于显示进度条
    for imgs, path in tqdm(data_loader_test):
        X = imgs
        outputs = model(X)
        
        # probability表示是否个对象的概率
        probability, pred = torch.max(F.softmax(outputs, dim=1).detach(), dim=1)
        
        batch_results = [(path_.item(), str(round(probability_.item()*100, 2))+'%', pred_.item())
                         for path_, probability_, pred_ in zip(path, probability, pred)]
        resultsx3 += batch_results
test()
