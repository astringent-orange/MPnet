'''
测试val代码，用于测试val代码是否正确
'''

import os
import numpy as np
from src.utils.opts import opts
from src.data.datasets import MPDataset

if __name__ == '__main__':
    opt = opts().parse()
    save_dir = getattr(opt, 'save_dir', './experiments/test_eval')
    dataset = MPDataset(opt, dataset_type='val')
    num_imgs = len(dataset)
    results = {}
    # 随机生成预测框，格式与run_eval一致
    for img_id in range(num_imgs):
        # 随机生成1~3个预测框
        n = np.random.randint(1, 4)
        boxes = []
        for _ in range(n):
            x1, y1 = np.random.uniform(0, 400, 2)
            x2, y2 = x1 + np.random.uniform(10, 100), y1 + np.random.uniform(10, 100)
            score = np.random.uniform(0.5, 1.0)
            obj_id = np.random.randint(0, 10)
            boxes.append([x1, y1, x2, y2, score, obj_id])
        results[img_id] = {1: np.array(boxes)}  # 只测类别1
    # 调用评测
    stats, _ = dataset.run_eval(results, save_dir, 'test')
    print('Test eval stats:', stats) 