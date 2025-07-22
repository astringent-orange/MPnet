'''
测试数据集dataset能否正常读取
'''

import argparse
from src.data.datasets import MPDataset
from src.utils.opts import opts
from src.utils.tools import print_banner

if __name__ == '__main__':
    opt = opts()
    dataset = MPDataset(opt, type='train')

    print_banner("print sample")

    for i in range(2):
        sample = dataset[i]
        print(f"样本{i}: video={sample['video']}, frame_id={sample['frame_id']}, "
              f"input.shape={sample['input'].shape}, "
              f"targets lens={[len(t) for t in sample['targets']]}, "
              f"img_path={sample['img_path']}")
        print(f"targets: {sample['targets']}")
        print()  # 每个输出结果之间换行
