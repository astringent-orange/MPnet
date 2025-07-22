'''
训练模型
'''

import os
from src.utils.opts import opts
from src.utils.tools import print_banner

import torch
from torch.utils.data import DataLoader

# 导入数据集
from src.data.datasets import MPDataset

from src.models.model import Model
from src.engine.trainer import Trainer
from src.utils.logger import Logger

def train(opt):
    # ********************** 准备训练 **********************
    print_banner('Preparing training')

    # 设置随机种子
    torch.manual_seed(opt.seed)
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    print(f'Using device: {opt.device}')

    # 创建数据集
    train_dataset = MPDataset(opt, type='train')
    val_dataset = MPDataset(opt, type='val')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    # 创建模型
    print('Create model')
    model = Model(opt)
    print(model)
    model.to(opt.device)

    # 创建优化器
    print('Create optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    print(optimizer)

    # 创建损失函数
    print('Create loss')

    # 创建训练器
    print('Create trainer')
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus[0], opt.device)

    # 创建保存目录
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # 创建记录器
    print('Create logger')
    logger = Logger(opt)



    # ********************** 开始训练 **********************
    print_banner('Start training')
    start_epoch = 0

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        pass

    # ********************** 结束训练 **********************
    print_banner('Training finished')
    logger.close()


if __name__ == "__main__":
    opt = opts().parse()
    train(opt)