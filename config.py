class Config:
    # 文件路径
    # 数据集根目录
    data_root = r'E:\datas'
    # 训练集存放路径
    data_train_root = r'E:\datas\train'
    # 验证集存放路径
    data_valid_root = r'E:\datas\valid'
    # 测试集存放路径
    data_test_root =r'E:\datas\test'

    # 常用参数
    # 图片大小
    input_size = 224
    # batch size
    batch_size = 10
    # mean and std
    # 通过抽样计算得到图片的均值mean和标准差std
    # mean = [0.470, 0.431, 0.393]
    # std = [0.274, 0.263, 0.260]
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]


