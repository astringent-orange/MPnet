import argparse
import os
import sys
from datetime import datetime

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basic 基本参数
        self.parser.add_argument('--down_ratio', type=int, default=1,
                                 help='output stride. Currently only supports for 1.')

        # train 训练参数
        self.parser.add_argument('--lr', type=float, default=1.25e-4,
                                 help='learning rate for batch size 4.')
        self.parser.add_argument('--lr_step', type=str, default='14',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=50,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=4,
                                 help='batch size')
        self.parser.add_argument('--val_intervals', type=int, default=10,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--seq_len', type=int, default=5,
                                 help='number of images for per sample. Currently supports 5.')

        # test 测试参数
        self.parser.add_argument('--K', type=int, default=450,
                                 help='max number of output objects.')

        # model 模型参数

        # dataset 数据集参数
        self.parser.add_argument('--dataroot', type=str, default='./data/dataset', 
                                help='path to dataset')

        # save 保存参数

    
    def parse(self):
        opt = self.parser.parse_args()


        print('\n====print opt====\n')
        print(opt)
        return opt
    
