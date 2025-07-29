'''
数据集类
'''

import os
import torch
import numpy as np
import cv2
import math
import datetime
import motmetrics as mm
from torch.utils.data import Dataset
from collections import defaultdict
from src.utils.opts import opts
from src.utils.tools import print_banner
from src.utils.image import (
    draw_umich_gaussian, gaussian_radius, flip, color_aug, 
    get_affine_transform, affine_transform, draw_msra_gaussian,
    draw_dense_reg
)
from src.utils.augmentations import Augmentation


class MPDataset(Dataset):
    def __init__(self, opt, dataset_type='train'):
        self.opt = opt

        self.dataset_type = dataset_type
        self.dataroot = os.path.join(opt.dataroot, dataset_type)
        self.images_dir = os.path.join(self.dataroot, 'images')  # 图片目录
        self.label_dir = os.path.join(self.dataroot, 'labels')   # 标签目录
        self.images = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.jpg')])    # 图片文件列表
        self.labels = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.txt')])     # 标签文件列表

        self.num_samples = len(self.images)
        self.seq_len = opt.seq_len
        self.max_objs = opt.max_objs
        self.down_ratio = opt.down_ratio
        self.num_classes = opt.num_classes

        # 图片归一化参数
        self.mean = np.array([0.49965, 0.49965, 0.49965], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.08255, 0.08255, 0.08255], dtype=np.float32).reshape(1, 1, 3)

        # 训练集使用数据增强
        if(self.dataset_type=='train'):
            self.aug = Augmentation(opt)
        else:
            self.aug = None

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

    def _coco_box_to_bbox(self, box):
        """
        将边界框格式的[x, y, w, h]转为[x1, y1, x2, y2]。
        """
        if len(box) == 0:
            return box
        else:
            box = np.array(box, dtype=np.int32)
            bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            return bbox

    def _get_border(self, border, size):
        """
        计算边界，用于数据增强。
        """
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _get_transoutput(self, c, s, height, width):
        """
        计算仿射变换矩阵，用于标签映射。
        """
        trans_output = get_affine_transform(c, s, 0, [width, height])
        return trans_output

    def transform_box(self, bbox, transform, output_w, output_h):
        """
        对bbox做仿射变换，得到中心点和高斯半径，用于生成heatmap。
        参数：
            bbox: 边界框[x1, y1, x2, y2]
            transform: 仿射变换矩阵
            output_w, output_h: 输出特征图宽高
        返回：
            bbox: 变换后的bbox
            ct: 中心点
            radius: 高斯半径
        """
        bbox[:2] = affine_transform(bbox[:2], transform)
        bbox[2:] = affine_transform(bbox[2:], transform)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        h = np.clip(h, 0, output_h - 1)
        w = np.clip(w, 0, output_w - 1)
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array(
            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct[0] = np.clip(ct[0], 0, output_w - 1)
        ct[1] = np.clip(ct[1], 0, output_h - 1)
        return bbox, ct, radius   

    def __getitem__(self, idx):
        """
        返回一个时序样本：
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
        # 文件名
        file_name = self.images[idx]

        # 解析文件名，获取帧号和视频编号
        imIdex = file_name.split('.')[0].split('_')[-1]         #000001
        imf = file_name.split('_')[0]                           #001
        imtype = '.'+file_name.split('.')[-1]                   #.jpg
        # 读取当前帧图片
        im0 = cv2.imread(os.path.join(self.images_dir, file_name))
        # 初始化时序图片数组 (H, W, 3, seq_len)
        img = np.zeros([im0.shape[0], im0.shape[1], 3, self.seq_len])
        pre_revrs_ids = []        # 存储前序帧的图片ID
        interval = []
        temp_id = idx

        # 读取时序帧图片，并做归一化
        for ii in range(self.seq_len):
            # 计算当前时序帧的帧号（保证最小为1，防止越界），格式化为6位字符串
            imIndexNew = '%06d' % max(int(imIdex) - self.seq_len + ii + 1, 1)
            # 构建当前时序帧的完整文件路径（如 '001/img1/000001.jpg'）
            imName = imf + '_' + imIndexNew + imtype
            # 仅当当前帧索引未超过实际帧号时，更新temp_id为对应的图片ID
            if ii <= int(imIdex) - 1:
                temp_id = idx - ii
            # 除了第0帧外，记录前序帧的图片ID和与当前帧的间隔
            if ii != 0:
                pre_revrs_ids.append(temp_id)
                interval.append(idx - temp_id)
            # 读取当前时序帧图片（BGR格式，H×W×3）
            im = cv2.imread(os.path.join(self.images_dir, imName))
            # 将图片像素归一化到[0,1]，再做均值方差归一化
            inp_i = (im.astype(np.float32) / 255.)
            inp_i = (inp_i - self.mean) / self.std
            # 存入时序图片数组的第ii帧（H, W, 3, seq_len）
            img[:, :, :, ii] = inp_i
     
        bbox_tol = []   # 当前帧所有目标的bbox
        cls_id_tol = [] # 当前帧所有目标的类别
        ids_tol = []    # 当前帧所有目标的ID

        with open(os.path.join(self.label_dir, self.labels[idx]), 'r') as f:
            for id, line in enumerate(f.readlines()):
                if id > self.max_objs:
                    break
                line = line.strip().split(' ')
                bbox_tol.append(self._coco_box_to_bbox(line[2:6]))  # 边界框
                cls_id_tol.append(int(line[6]))                     # 类别ID
                ids_tol.append(int(line[1]))                        # 目标ID

        # 获取前序帧的目标框和ID
        pre_bboxes = defaultdict(list)
        pre_ids = defaultdict(list)
        for i in range(self.seq_len - 1):
            pre_ann_dir = os.path.join(self.label_dir, self.labels[idx-self.seq_len+i+1])
            with open(pre_ann_dir, 'r') as f:
                for id, line in enumerate(f.readlines()):
                    if id > self.max_objs:
                        break
                    line = line.strip().split(' ')
                    pre_bboxes[i + 1].append(self._coco_box_to_bbox(line[2:6]))
                    pre_ids[i + 1].append(int(line[1]))

        # 数据增强（仅训练集）
        if self.aug is not None:
            bbox_tol = np.array(bbox_tol)
            cls_id_tol = np.array(cls_id_tol)
            ids_tol = np.array(ids_tol)
            for i in range(self.seq_len - 1):
                pre_bboxes[i + 1] = np.array(pre_bboxes[i + 1])
                pre_ids[i + 1] = np.array(pre_ids[i + 1])
            # 增强图片和标签
            img, _, bbox_tol, cls_id_tol, ids_tol, pre_bboxes, pre_ids = self.aug(img, img, bbox_tol, cls_id_tol, ids_tol, pre_bboxes, pre_ids)
            bbox_tol = bbox_tol.tolist()
            cls_id_tol = cls_id_tol.tolist()
            ids_tol = ids_tol.tolist()
            for i in range(self.seq_len - 1):
                pre_bboxes[i + 1] = pre_bboxes[i + 1].tolist()
                pre_ids[i + 1] = pre_ids[i + 1].tolist()
        
        # 确保 num_objs 在所有情况下都被定义
        num_objs = len(bbox_tol)

        # 转换图片shape为 (seq_len, 3, H, W)
        inp = img.transpose(3, 2, 0, 1).astype(np.float32)
        # 裁剪图片尺寸为16的倍数
        height, width = img.shape[0] - img.shape[0] % 32, img.shape[1] - img.shape[1] % 32
        inp = inp[:, :, 0:height, 0:width]
        # 计算中心点和缩放因子
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(width, height) * 1.0
        ret = {'input': inp}

        down_ratios = [1]
        for ratio in down_ratios:
            # 计算输出特征图尺寸
            output_h = height // ratio // self.down_ratio
            output_w = width // ratio // self.down_ratio
            trans_output = self._get_transoutput(c, s, output_h, output_w)

            # 初始化标签（heatmap、wh、reg、mask等）
            hm = np.zeros((self.seq_len, self.num_classes, output_h, output_w), dtype=np.float32)      # 目标类别热力图 (时序帧数, 类别数, H, W)
            hm_seq = np.zeros((self.seq_len, 1, output_h, output_w), dtype=np.float32)                 # 目标存在性热力图 (时序帧数, 1, H, W)
            wh = np.zeros((self.max_objs, 2), dtype=np.float32)                                   # 目标宽高 (最大目标数, 2)
            reg = np.zeros((self.max_objs, 2), dtype=np.float32)                                  # 目标中心点偏移 (最大目标数, 2)
            ind = np.zeros((self.max_objs), dtype=np.int64)                                       # 目标中心点在特征图上的索引 (最大目标数)
            reg_mask = np.zeros((self.max_objs), dtype=np.uint8)                                  # 目标mask, 有效目标为1 (最大目标数)

            ind_dis = np.zeros((self.seq_len - 1, self.max_objs), dtype=np.int64)                      # 前序帧目标中心点索引 (帧数-1, 最大目标数)
            dis_mask = np.zeros((self.seq_len - 1, self.max_objs), dtype=np.uint8)                     # 前序帧目标mask, 有效目标为1 (帧数-1, 最大目标数)
            dis = np.zeros((self.seq_len - 1, self.max_objs, 2), dtype=np.float32)                     # 前序帧目标中心点位移 (帧数-1, 最大目标数, 2)

            gt_det = []
            # 遍历每个目标，生成标签
            for k in range(num_objs):
                bbox = bbox_tol[k]
                cls_id = cls_id_tol[k] - 1
                obj_id = ids_tol[k]
                # 仿射变换bbox到输出空间
                bbox[:2] = affine_transform(bbox[:2], trans_output)
                bbox[2:] = affine_transform(bbox[2:], trans_output)

                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                h = np.clip(h, 0, output_h - 1)
                w = np.clip(w, 0, output_w - 1)
                if h > 0 and w > 0:
                    # 处理时序帧的标签
                    for i in range(1, self.seq_len):
                        if i != self.seq_len - 1:
                            if pre_ids[i].count(obj_id) != 0 and pre_ids[i + 1].count(obj_id) != 0:
                                temp_ind_pre = pre_ids[i].index(obj_id)
                                temp_ind_later = pre_ids[i + 1].index(obj_id)
                                temp_box_pre = pre_bboxes[i][temp_ind_pre]
                                temp_box_later = pre_bboxes[i + 1][temp_ind_later]

                                temp_box_pre, ct_pre, radius_pre = self.transform_box(temp_box_pre, trans_output, output_w, output_h)
                                temp_box_later, ct_later, radius_later = self.transform_box(temp_box_later, trans_output, output_w, output_h)
                                
                                ct_int_pre = ct_pre.astype(np.int32)
                                draw_umich_gaussian(hm[i - 1][cls_id], ct_int_pre, radius_pre)
                                draw_umich_gaussian(hm_seq[i - 1][0], ct_int_pre, radius_pre)
                                ind_dis[i - 1][temp_ind_pre] = ct_int_pre[1] * output_w + ct_int_pre[0]
                                dis[i - 1][temp_ind_pre] = ct_later - ct_pre
                                dis_mask[i - 1][temp_ind_pre] = 1
                        else:
                            if pre_ids[i].count(obj_id) != 0:
                                temp_ind_pre = pre_ids[i].index(obj_id)
                                temp_box_pre = pre_bboxes[i][temp_ind_pre]
                                temp_box_later = bbox_tol[k]

                                temp_box_pre, ct_pre, radius_pre = self.transform_box(temp_box_pre, trans_output, output_w, output_h)
                                temp_box_later, ct_later, radius_later = self.transform_box(temp_box_later, trans_output, output_w, output_h)
                                
                                ct_int_pre = ct_pre.astype(np.int32)
                                draw_umich_gaussian(hm[i - 1][cls_id], ct_int_pre, radius_pre)
                                draw_umich_gaussian(hm_seq[i - 1][0], ct_int_pre, radius_pre)
                                ind_dis[i - 1][temp_ind_pre] = ct_int_pre[1] * output_w + ct_int_pre[0]
                                dis[i - 1][temp_ind_pre] = ct_later - ct_pre
                                dis_mask[i - 1][temp_ind_pre] = 1

                    # 处理当前帧的标签
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    ct = np.array(
                        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct[0] = np.clip(ct[0], 0, output_w - 1)
                    ct[1] = np.clip(ct[1], 0, output_h - 1)
                    ct_int = ct.astype(np.int32)
                    draw_umich_gaussian(hm[self.seq_len - 1][cls_id], ct_int, radius)
                    draw_umich_gaussian(hm_seq[self.seq_len - 1][0], ct_int, radius)
                    wh[k] = 1. * w, 1. * h
                    ind[k] = ct_int[1] * output_w + ct_int[0]
                    reg[k] = ct - ct_int
                    reg_mask[k] = 1
                    gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                                ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
            # 保存所有标签到ret
            ret[ratio] = {'hm': hm, 'hm_seq': hm_seq, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'dis_ind': ind_dis, 'dis': dis, 'dis_mask': dis_mask}

        # 填充空目标
        for kkk in range(num_objs, self.max_objs):
            bbox_tol.append([])

        ret['file_name'] = file_name

        # 返回图片ID和标签字典
        return idx, ret 
        
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
        使用motmetrics库评测：输出MOTA、IDF1等，并返回。
        """
        acc = mm.MOTAccumulator(auto_id=True)
        
        # 遍历每张图片（帧）
        for img_id, dets in results.items():
            # 只考虑类别1（单类别）
            pred_boxes = dets.get(1, np.zeros((0, 6)))
            pred_boxes = pred_boxes if isinstance(pred_boxes, np.ndarray) else np.array(pred_boxes)
            
            # 读取GT
            fname = self.images[img_id]
            base = os.path.splitext(fname)[0]
            label_path = os.path.join(self.label_dir, base + '.txt')
            gt_objs = self._read_label(label_path)
            gt_ids = [str(obj['obj_id']) for obj in gt_objs]
            gt_boxes = [obj['bbox'] for obj in gt_objs]
            
            # 预测ID和框 - 修正处理逻辑
            if pred_boxes.shape[0] > 0 and pred_boxes.shape[1] >= 6:
                pred_ids = [str(int(box[5])) for box in pred_boxes]
                pred_boxes_xyxy = [box[:4] for box in pred_boxes]
            else:
                pred_ids = []
                pred_boxes_xyxy = []
            
            # 计算距离矩阵（1 - IoU）
            if len(gt_boxes) > 0 and len(pred_boxes_xyxy) > 0:
                # 确保gt_boxes和pred_boxes_xyxy都是正确的格式
                gt_boxes_array = np.array(gt_boxes).reshape(-1, 4)
                pred_boxes_array = np.array(pred_boxes_xyxy).reshape(-1, 4)
                
                dists = mm.distances.iou_matrix(
                    gt_boxes_array, pred_boxes_array, max_iou=0.5
                )
                dists = 1 - dists  # motmetrics的距离是1-IOU
            else:
                # 当没有GT或没有预测时，创建正确形状的距离矩阵
                dists = np.full((len(gt_boxes), len(pred_boxes_xyxy)), np.nan)
            
            # # 调试信息
            # if img_id < 5:  # 只打印前5帧的调试信息
            #     print(f'img_id={img_id}, gt_ids={len(gt_ids)}, pred_ids={len(pred_ids)}, dists.shape={dists.shape}')
            #     print(f'gt_boxes: {len(gt_boxes)}, pred_boxes_xyxy: {len(pred_boxes_xyxy)}')
            
            # 累加到accumulator
            acc.update(gt_ids, pred_ids, dists)
        
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
        stats = {k: summary.loc['acc', k] for k in summary.columns}
        
        # 保存结果
        os.makedirs(save_dir, exist_ok=True)
        txt_path = os.path.join(save_dir, 'eval_results.txt')
        with open(txt_path, 'a') as f:
            f.write(f'==== Eval at {time_str} ====\n')
            f.write(mm.io.render_summary(
                summary,
                formatters=mh.formatters,
                namemap=mm.io.motchallenge_metric_names
            ))
            f.write('\n')
        return stats, None



        