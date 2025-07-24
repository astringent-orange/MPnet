'''
测试数据集dataset能否正常读取
'''

import argparse
from src.data.datasets import MPDataset
from src.utils.opts import opts
from src.utils.tools import print_banner

import numpy as np

def print_dict_structure(d, indent=0):
    prefix = '  ' * indent
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}: dict")
                print_dict_structure(v, indent+1)
            elif isinstance(v, np.ndarray):
                print(f"{prefix}{k}: np.ndarray, shape={v.shape}, dtype={v.dtype}")
            elif hasattr(v, 'shape') and hasattr(v, 'dtype'):
                print(f"{prefix}{k}: tensor, shape={tuple(v.shape)}, dtype={v.dtype}")
            else:
                print(f"{prefix}{k}: {type(v).__name__}, value={v}")
    else:
        print(f"{prefix}{type(d).__name__}: {d}")

if __name__ == '__main__':
    opt = opts().parse()
    dataset = MPDataset(opt, dataset_type='train')

    print_banner("print sample")

    for i in range(2):
        img_id, sample = dataset[i]
        print(f"样本{i}: img_id={img_id}, file_name={sample['file_name']}, input.shape={sample['input'].shape}")
        print("ret结构:")
        print_dict_structure(sample)
        print()  # 每个输出结果之间换行
