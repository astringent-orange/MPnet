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


def train(opt):
    # 设置随机种子
    torch.manual_seed(opt.seed)
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    print_banner(f'Using device: {opt.device}')

    # 创建数据集
    train_dataset = MPDataset(opt, type='train')
    val_dataset = MPDataset(opt, type='val')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    # 创建模型
    print_banner('Create model')
    model = Model(opt)
    print(model)
    model.to(opt.device)

    # 创建优化器
    print_banner('Create optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # 创建损失函数

    # 创建学习率调度器
    print_banner('Create scheduler')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)




if __name__ == "__main__":
    opt = opts().parse()
    train(opt)