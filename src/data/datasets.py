'''
数据集类
'''

import os
import torch
import numpy as np
import cv2
import math
import datetime
from torch.utils.data import Dataset
from collections import defaultdict
from src.utils.opts import opts
from src.utils.tools import print_banner
from src.utils.image import draw_umich_gaussian, gaussian_radius


def _normalize_img(img, mean, std):
    img = img.astype(np.float32) / 255.
    return (img - mean) / std

def _load_img(img_path, img_size):
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, img_size)
    return img

def _zero_img(img_size):
    return np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)

class MPDataset(Dataset):
    """
    MPDataset：适配所有帧图片都在同一目录，文件名格式为[videoid]_[frameid].jpg的数据集。
    支持多类别目标检测与跟踪任务。
    返回内容与COCO类一致，便于与现有检测/跟踪训练流程兼容。
    """
    def __init__(self, opt, dataset_type='train'):
        self.opt = opt
        self.dataset_type = dataset_type
        self.dataroot = os.path.join(opt.dataroot, dataset_type)
        self.images_dir = os.path.join(self.dataroot, 'images')  # 图片目录
        self.label_dir = os.path.join(self.dataroot, 'labels')   # 标签目录
        
        self.images = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.jpg')])
        self.labels = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.txt')])

        self.num_samples = len(self.images)
        self.seq_len = opt.seq_len
        self.max_objs = opt.max_objs
        self.down_ratio = getattr(opt, 'down_ratio', 4)
        self.num_classes = getattr(opt, 'num_classes', 1)
        if hasattr(opt, 'img_height') and hasattr(opt, 'img_width'):
            self.img_size = (opt.img_height, opt.img_width)
        elif hasattr(opt, 'img_size'):
            self.img_size = opt.img_size
        else:
            self.img_size = (512, 512)
        self.mean = np.array([0.49965, 0.49965, 0.49965], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.08255, 0.08255, 0.08255], dtype=np.float32).reshape(1, 1, 3)
        self.transform = getattr(opt, 'transform', None)

        print(f"Loaded {self.num_samples} {self.dataset_type} samples")

    def __len__(self):
        return self.num_samples

    def _read_label(self, label_path):
        """
        读取单帧的label.txt，返回目标列表。
        每行格式：帧号 物体id x y w h 类别 -1 -1 -1
        这里只取物体id, x, y, w, h, 类别。
        返回：list，每个元素为dict，含obj_id, bbox, cls
        """
        targets = []
        if not label_path or not os.path.isfile(label_path):
            return targets
        with open(label_path, 'r') as f:
            for line in f:
                items = line.strip().split()
                if len(items) < 7:
                    continue
                obj_id = int(items[1])
                x, y, w, h = map(float, items[2:6])
                cls = int(items[6])
                # bbox为[x1, y1, x2, y2]格式
                targets.append({
                    'obj_id': obj_id,
                    'bbox': [x, y, x + w, y + h],
                    'cls': cls
                })
        return targets

    def _sample_sequence(self, idx, cur_video_id):
        """
        采样时序帧，返回imgs, targets, file_names
        """
        imgs, targets, file_names = [], [], []
        for i in range(self.seq_len):
            tid = idx - self.seq_len + i + 1
            if tid < 0 or tid >= self.num_samples:
                imgs.append(_zero_img(self.img_size))
                targets.append([])
                file_names.append('')
                continue
            fname = self.images[tid]
            base = os.path.splitext(fname)[0]
            video_id = base.split('_', 1)[0] if '_' in base else 'unknown'
            if video_id != cur_video_id:
                imgs.append(_zero_img(self.img_size))
                targets.append([])
                file_names.append(fname)
                continue
            img_path = os.path.join(self.images_dir, fname)
            label_path = os.path.join(self.label_dir, base + '.txt') if self.label_dir else None
            img = _load_img(img_path, self.img_size)
            tars = self._read_label(label_path)
            imgs.append(img)
            targets.append(tars)
            file_names.append(fname)
        return imgs, targets, file_names

    def __getitem__(self, idx):
        """
        返回一个时序样本，内容与COCO类一致：
        img_id, ret
        其中：
        - img_id: 当前样本唯一id（此处用idx）
        - ret: 字典，包含：
            'input': 时序图片，shape=(seq_len, 3, H, W)，float32，已归一化
            1: ratio=1的标签字典，含：
                'hm': 时序heatmap，(seq_len, num_classes, H', W')
                'hm_seq': 时序单通道heatmap，(seq_len, 1, H', W')
                'reg_mask': (max_objs,)
                'ind': (max_objs,)
                'wh': (max_objs, 2)
                'reg': (max_objs, 2)
                'dis_ind': (seq_len-1, max_objs)
                'dis': (seq_len-1, max_objs, 2)
                'dis_mask': (seq_len-1, max_objs)
            'file_name': 当前帧图片文件名
        """
        cur_fname = self.images[idx]
        base = os.path.splitext(cur_fname)[0]
        cur_video_id = base.split('_', 1)[0] if '_' in base else 'unknown'
        imgs, targets, file_names = self._sample_sequence(idx, cur_video_id)
        # 归一化和变换
        imgs = [_normalize_img(img, self.mean, self.std) for img in imgs]
        if self.transform:
            imgs = [self.transform(img) for img in imgs]
        # 转换为PyTorch常用格式 (seq_len, 3, H, W)
        imgs = np.stack(imgs, axis=0).transpose(0, 3, 1, 2)
        height, width = self.img_size
        output_h, output_w = height // self.down_ratio, width // self.down_ratio
        seq_num, max_objs, num_classes = self.seq_len, self.max_objs, self.num_classes
        # 标签初始化
        hm = np.zeros((seq_num, num_classes, output_h, output_w), dtype=np.float32)  # 每帧每类heatmap
        hm_seq = np.zeros((seq_num, 1, output_h, output_w), dtype=np.float32)        # 每帧单通道heatmap
        wh = np.zeros((max_objs, 2), dtype=np.float32)      # 当前帧每目标宽高
        reg = np.zeros((max_objs, 2), dtype=np.float32)     # 当前帧每目标中心点偏移
        ind = np.zeros((max_objs), dtype=np.int64)          # 当前帧每目标中心点索引
        reg_mask = np.zeros((max_objs), dtype=np.uint8)     # 当前帧每目标mask
        ind_dis = np.zeros((seq_num - 1, max_objs), dtype=np.int64)      # 跟踪分支索引
        dis_mask = np.zeros((seq_num - 1, max_objs), dtype=np.uint8)     # 跟踪分支mask
        dis = np.zeros((seq_num - 1, max_objs, 2), dtype=np.float32)     # 跟踪分支位移
        gt_det = []
        # 统计所有目标id在各帧的中心点
        obj_id2ct = defaultdict(lambda: [None]*seq_num)
        obj_id2ind = defaultdict(lambda: [None]*seq_num)
        for i in range(seq_num):
            tars = targets[i]
            for k, tar in enumerate(tars):
                bbox = tar['bbox']
                cls_id = 0  # 单类别
                obj_id = tar['obj_id']
                # 缩放到输出特征图
                bbox = np.array(bbox, dtype=np.float32)
                bbox = bbox * [output_w/width, output_h/height, output_w/width, output_h/height]
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    # 计算高斯半径
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    # 计算中心点
                    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct[0] = np.clip(ct[0], 0, output_w - 1)
                    ct[1] = np.clip(ct[1], 0, output_h - 1)
                    ct_int = ct.astype(np.int32)
                    # 画高斯热力图
                    draw_umich_gaussian(hm[i][cls_id], ct_int, radius)
                    draw_umich_gaussian(hm_seq[i][0], ct_int, radius)
                    # 只在最后一帧填充检测标签
                    if i == seq_num - 1:
                        if k < max_objs:
                            wh[k] = 1. * w, 1. * h
                            ind[k] = ct_int[1] * output_w + ct_int[0]
                            reg[k] = ct - ct_int
                            reg_mask[k] = 1
                            gt_det.append([ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
                    obj_id2ct[obj_id][i] = ct
                    obj_id2ind[obj_id][i] = ct_int[1] * output_w + ct_int[0]
        # 生成dis标签（跟踪分支）
        for k in range(max_objs):
            for i in range(seq_num - 1):
                # 只处理当前帧有目标且前一帧也有同id目标的情况
                for obj_id in obj_id2ct:
                    if obj_id2ct[obj_id][i] is not None and obj_id2ct[obj_id][i+1] is not None:
                        ind_dis[i][k] = obj_id2ind[obj_id][i]
                        dis[i][k] = obj_id2ct[obj_id][i+1] - obj_id2ct[obj_id][i]
                        dis_mask[i][k] = 1
        # 填充空目标（wh, reg, ind, reg_mask已初始化为0）
        for kkk in range(len(targets[seq_num-1]), max_objs):
            pass
        # 构造ret字典
        ret = {
            'input': imgs.astype(np.float32),
            1: {
                'hm': hm,
                'hm_seq': hm_seq,
                'reg_mask': reg_mask,
                'ind': ind,
                'wh': wh,
                'reg': reg,
                'dis_ind': ind_dis,
                'dis': dis,
                'dis_mask': dis_mask
            },
            'file_name': cur_fname
        }
        img_id = idx
        return img_id, ret 
        
    def save_results(self, results, save_dir, time_str, eval_stats=None):
        """
        保存检测结果和评测统计到save_dir/eval_results.txt。
        results: 检测结果
        eval_stats: dict, 评测统计（如mAP50、precision等），可为None
        """
        os.makedirs(save_dir, exist_ok=True)
        txt_path = os.path.join(save_dir, 'eval_results.txt')
        with open(txt_path, 'a') as f:
            f.write(f'==== Eval at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ({time_str}) ====' + '\n')
            for img_id, dets in results.items():
                f.write(f'Image {img_id}:\n')
                for cls_id, boxes in dets.items():
                    for box in boxes:
                        f.write(f'  class={cls_id}, box={box.tolist()}\n')
            if eval_stats is not None:
                f.write('Eval stats:\n')
                for k, v in eval_stats.items():
                    f.write(f'  {k}: {v}\n')
            f.write('\n')

    def run_eval(self, results, save_dir, time_str):
        """
        评测：输出mAP50、准确率、召回率、MOTA、IDF1，并保存到txt。
        """
        # IoU计算函数
        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
            boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
            iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)
            return iou

        iou_thr = 0.5
        all_TP, all_FP, all_FN = 0, 0, 0
        all_precisions, all_recalls, all_ap = [], [], []
        all_mota_num, all_mota_den, all_idtp, all_idfp, all_idfn = 0, 0, 0, 0, 0
        # 遍历每张图片
        for img_id, dets in results.items():
            # 只考虑类别1（单类别）
            pred_boxes = dets.get(1, np.zeros((0, 6)))
            pred_boxes = pred_boxes if isinstance(pred_boxes, np.ndarray) else np.array(pred_boxes)
            pred_boxes = pred_boxes.tolist() if isinstance(pred_boxes, np.ndarray) else pred_boxes
            # 读取GT
            fname = self.images[img_id]
            base = os.path.splitext(fname)[0]
            label_path = os.path.join(self.label_dir, base + '.txt')
            gt_objs = self._read_label(label_path)
            gt_boxes = [obj['bbox'] for obj in gt_objs]
            gt_ids = [obj['obj_id'] for obj in gt_objs]
            matched_gt = set()
            TP, FP, FN = 0, 0, 0
            idtp, idfp, idfn = 0, 0, 0
            # 匹配预测框和GT框
            for pred in pred_boxes:
                pred_box = pred[:4]
                pred_score = pred[4]
                best_iou = 0
                best_gt_idx = -1
                for idx, gt_box in enumerate(gt_boxes):
                    if idx in matched_gt:
                        continue
                    iou_val = iou(pred_box, gt_box)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = idx
                if best_iou >= iou_thr:
                    TP += 1
                    matched_gt.add(best_gt_idx)
                    # IDF1/MOTA简单实现：如果obj_id一致则idtp+=1，否则idfp+=1
                    if 'obj_id' in gt_objs[best_gt_idx] and len(pred) > 5 and int(pred[5]) == int(gt_objs[best_gt_idx]['obj_id']):
                        idtp += 1
                    else:
                        idfp += 1
                else:
                    FP += 1
                    idfp += 1
            FN = len(gt_boxes) - len(matched_gt)
            idfn = FN
            all_TP += TP
            all_FP += FP
            all_FN += FN
            all_mota_num += (FP + FN + idfp)  # 近似MOTA分母
            all_mota_den += (len(gt_boxes) if len(gt_boxes) > 0 else 1)
            all_idtp += idtp
            all_idfp += idfp
            all_idfn += idfn
            precision = TP / (TP + FP + 1e-6)
            recall = TP / (TP + FN + 1e-6)
            all_precisions.append(precision)
            all_recalls.append(recall)
            # AP50: 只用单阈值，等价于precision@recall
            all_ap.append(precision)
        # 统计整体指标
        precision = np.mean(all_precisions) if all_precisions else 0
        recall = np.mean(all_recalls) if all_recalls else 0
        map50 = np.mean(all_ap) if all_ap else 0
        mota = 1 - (all_mota_num / (all_mota_den + 1e-6)) if all_mota_den > 0 else 0
        idf1 = (2 * all_idtp) / (2 * all_idtp + all_idfp + all_idfn + 1e-6) if (2 * all_idtp + all_idfp + all_idfn) > 0 else 0
        stats = {'map50': map50, 'precision': precision, 'recall': recall, 'MOTA': mota, 'IDF1': idf1}
        # 保存结果和评测统计
        self.save_results(results, save_dir, time_str, eval_stats=stats)
        return stats, None



        