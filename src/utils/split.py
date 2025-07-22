'''
将训练集按照8:2的比例划分为训练集和验证集
'''
import os
import shutil
import random
from collections import defaultdict

# 配置路径
TRAIN_IMAGES_DIR = 'data/dataset/train/images'
TRAIN_LABELS_DIR = 'data/dataset/train/labels'
VAL_IMAGES_DIR = 'data/dataset/val/images'
VAL_LABELS_DIR = 'data/dataset/val/labels'

# 验证集比例
VAL_RATIO = 0.2  # 可根据需要调整


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_video_id(filename):
    # 假设文件名格式为 videoID_frameID.jpg，如 1-2_000000.jpg
    return filename.split('_')[0]


def split_by_video():
    ensure_dir(VAL_IMAGES_DIR)
    ensure_dir(VAL_LABELS_DIR)

    # 收集所有图片，按视频ID分组
    image_files = [f for f in os.listdir(TRAIN_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    video_to_images = defaultdict(list)
    for img in image_files:
        video_id = get_video_id(img)
        video_to_images[video_id].append(img)

    video_ids = list(video_to_images.keys())
    val_video_count = max(1, int(len(video_ids) * VAL_RATIO))
    val_video_ids = set(random.sample(video_ids, val_video_count))

    val_count = 0
    for video_id in val_video_ids:
        for img in video_to_images[video_id]:
            # 移动图片
            src_img = os.path.join(TRAIN_IMAGES_DIR, img)
            dst_img = os.path.join(VAL_IMAGES_DIR, img)
            shutil.move(src_img, dst_img)
            # 移动对应标签
            label_file = os.path.splitext(img)[0] + '.txt'
            src_label = os.path.join(TRAIN_LABELS_DIR, label_file)
            dst_label = os.path.join(VAL_LABELS_DIR, label_file)
            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)
            else:
                print(f'警告: 未找到标签文件 {label_file}')
            val_count += 1

    print(f"已将{val_count}张图片（{len(val_video_ids)}个视频）划分为验证集。")


if __name__ == '__main__':
    split_by_video() 