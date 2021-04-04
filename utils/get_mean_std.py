import torch
import numpy as np

def get_mean_std(data_images):
    '''
    :param data_images: 加载好的数据集
    :return: mean,std
    '''
    times, mean, std = 0, 0, 0
    data_loader = {x: torch.utils.data.DataLoader(dataset=data_images[x],
                                                  batch_size=1000,
                                                  shuffle=True)
                          for x in ['train', 'valid']}
    for imgs, labels in data_loader['train']:
        # imgs.shape = torch.Size([32, 3, 64, 64])
        times += 1
        mean += np.mean(imgs.numpy(), axis=(0, 2, 3))
        std += np.std(imgs.numpy(), axis=(0, 2, 3))
        print('times:', times)

    mean /= times
    std /= times
    return mean, std
