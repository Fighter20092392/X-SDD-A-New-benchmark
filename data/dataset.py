import torch
from torchvision import datasets, transforms
import os
from config import Config
from PIL import Image

# 调用配置文件
conf = Config()

class Dataset:
    def __init__(self, train=True):
        # 图片预处理
        # Compose用于将多个transfrom组合起来
        # ToTensor()将像素从[0, 255]转换为[0, 1.0]
        # Normalize()用均值和标准差对图像标准化处理 x'=(x-mean)/std，加速收敛
        self.transform = transforms.Compose([transforms.Resize((conf.input_size, conf.input_size)),
                                             #transforms.Resize(224),
                                             #transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
                                             #transforms.RandomHorizontalFlip(),
                                             #transforms.RandomVerticalFlip(),
                                             transforms.ToTensor(),
                                     
                                             transforms.Normalize(conf.mean, conf.std)
                                             ])
        self.train = train
        # 加载训练数据集和验证数据集
        if train:

            # 数据加载
            # 这里使用通用的ImageFolder和DataLoader数据加载器
            # 数据类型 data_images = {'train': xxx, 'valid': xxx}
            self.data_images = {x: datasets.ImageFolder(root=os.path.join(conf.data_root, x),
                                                        transform=self.transform)
                                for x in ['train', 'valid']}
            self.data_images_loader = {x: torch.utils.data.DataLoader(dataset=self.data_images[x],
                                                                      batch_size=conf.batch_size,
                                                                      shuffle=True)
                                       for x in ['train', 'valid']}
            # 图片分类 ['cat', 'dog']
            self.classes = self.data_images['train'].classes
            # 图片分类键值对 {'cat': 0, 'dog': 1}
            self.classes_index = self.data_images['train'].class_to_idx

        # 加载测试数据集
        else:
            images = [os.path.join(conf.data_test_root, img) for img in os.listdir(conf.data_test_root)]
            self.images=images
#self.images = sorted(images, key=lambda x: int(x.split('.')[-2].split('/')[-1]))

    # 重载专有方法__getitem__
    def __getitem__(self, index):
        img_path = self.images[index]
        label = int(self.images[index].split('.')[-2].split('\\')[-1])
        data_images_test = Image.open(img_path).convert('RGB')
        data_images_test = self.transform(data_images_test)
        return data_images_test, label

    # 重载专有方法__len__
    def __len__(self):
        return len(self.images)




