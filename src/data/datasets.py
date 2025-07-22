'''
数据集类
'''

import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from src.utils.opts import opts
from src.utils.tools import print_banner

class MPDataset(Dataset):
    def __init__(self, opt, type='train'):
        super(MPDataset, self).__init__()
        self.opt = opt          # 获取配置文件
        self.type = type        # 获取数据集类型

        print_banner(f'Initializing {self.type} Dataset')

        self.root_dir = self.opt.dataroot
        self.image_dir = os.path.join(self.root_dir, self.type, 'images') # 获取数据集路径
        self.label_dir = os.path.join(self.root_dir, self.type, 'labels') # 获取标签集路径
        self.image_list = os.listdir(self.image_dir) # 获取数据集列表
        self.label_list = os.listdir(self.label_dir) # 获取标签集列表
        self.image_list.sort()  # 对图片列表进行排序，确保顺序一致
        self.label_list.sort()  # 对标签列表进行排序，确保顺序一致

        self.num_samples = len(self.image_list) # 获取数据集长度

        # img_size表示输入图片的尺寸，格式为(height, width)
        # hasattr(obj, 'attr') 用于判断对象obj是否有名为'attr'的属性
        if hasattr(self.opt, 'img_height') and hasattr(self.opt, 'img_width'):      # 优先使用img_height和img_width
            self.img_size = (self.opt.img_height, self.opt.img_width)
        elif hasattr(self.opt, 'img_size'):                                         # 其次使用img_size
            self.img_size = self.opt.img_size
        else:
            self.img_size = (512, 512)                                              # 否则默认(512, 512)

        self.down_ratio = self.opt.down_ratio    # 下采样比例
        self.max_objs   = self.opt.K             # 最大目标数
        self.seq_len     = self.opt.seq_len        # 时序长度
        self.transform = self.opt.transform if hasattr(self.opt, 'transform') else None

         # 图像归一化均值
        self.mean = np.array([0.49965, 0.49965, 0.49965],
                            dtype=np.float32).reshape(1, 1, 3)
        # 图像归一化方差
        self.std = np.array([0.08255, 0.08255, 0.08255],
                            dtype=np.float32).reshape(1, 1, 3)



        print(f'Loaded {self.num_samples} {self.type} images')

    def _read_label(self, label_path):
        """
        读取单帧的label.txt，返回目标列表。
        每行格式：帧号 物体id x y h w 类别1 -1 -1 -1
        这里只取物体id, x, y, h, w, 类别
        """
        targets = []
        if label_path is None or not os.path.isfile(label_path):
            return targets
        with open(label_path, 'r') as f:
            for line in f:
                items = line.strip().split()
                if len(items) < 7:
                    continue
                obj_id = int(items[1])
                x, y, h, w = map(float, items[2:6])
                cls = int(items[6])
                targets.append({
                    'obj_id': obj_id,
                    'bbox': [x, y, x + w, y + h],
                    'cls': cls
                })
        return targets

    def __len__(self):
        return self.num_samples # 返回数据集长度

    def __getitem__(self, idx):
        """
        返回一个时序样本，内容和格式如下：
            {
                'input': np.ndarray,  # shape=(seq_len, 3, H, W)，float32，时序图片张量
                'targets': list,     # 长度为seq_len的列表，每个元素为该帧的目标列表（每个目标为dict，含bbox、obj_id、cls）
                'video': str,        # 当前帧所属视频的ID（字符串）
                'frame_id': int,     # 当前帧的帧号（整数）
                'img_path': str      # 当前帧图片的完整路径
            }
        """
        # 当前帧的文件名
        cur_fname = self.image_list[idx]
        # 解析当前帧的video_id和frame_id
        # 使用os.path.splitext将文件名cur_fname分割为(主文件名, 扩展名)元组，这里取[0]即去除扩展名部分
        base = os.path.splitext(cur_fname)[0]
        if '_' in base:
            cur_video_id, cur_frame_id = base.split('_', 1)
        else:
            cur_video_id, cur_frame_id = 'unknown', -1
        imgs = []   # 存储时序帧图片
        targets = [] # 存储时序帧标签
        # 以当前帧为结尾，向前采样seq_len帧
        for i in range(self.seq_len):
            tid = idx - self.seq_len + i + 1  # 计算采样帧的索引
            if tid < 0 or tid >= len(self.image_list):
                # 边界情况：索引越界，用全0图像和空标签补齐
                img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                tars = []
            else:
                fname = self.image_list[tid]
                base = os.path.splitext(fname)[0]
                # 解析采样帧的video_id和frame_id
                if '_' in base:
                    video_id, frame_id = base.split('_', 1)
                else:
                    video_id, frame_id = 'unknown', -1
                # 判断是否与当前帧属于同一视频
                if video_id != cur_video_id:
                    # 跨视频则补零
                    img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                    tars = []
                else:
                    # 构建图片和标签路径
                    img_path = os.path.join(self.root_dir, self.type, 'images', fname)
                    label_path = os.path.join(self.root_dir, self.type, 'labels', base + '.txt') if self.label_dir else None
                    # 读取图片
                    img = cv2.imread(img_path)
                    if img is None:
                        img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                    else:
                        img = cv2.resize(img, self.img_size)
                    # 读取标签
                    tars = self._read_label(label_path)
            # 归一化处理
            img = img.astype(np.float32) / 255.
            img = (img - self.mean) / self.std
            imgs.append(img)
            targets.append(tars)
        # 可选数据增强
        if self.transform:
            imgs = [self.transform(img) for img in imgs]
        # 转换为PyTorch常用格式 (seq_len, 3, H, W)
        imgs = np.stack(imgs, axis=0).transpose(0, 3, 1, 2)
        # 构建当前帧的图片路径
        img_path = os.path.join(self.root_dir, cur_fname)
        # 返回结果字典
        return {
            'input': imgs.astype(np.float32),   # 时序图片张量
            'targets': targets,                 # 时序标签列表
            'video': cur_video_id,              # 当前帧所属视频
            'frame_id': int(cur_frame_id) if cur_frame_id != -1 else -1,  # 当前帧号
            'img_path': img_path                # 当前帧图片路径
        } 


        