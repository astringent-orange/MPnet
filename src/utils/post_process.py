from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from src.utils.image import transform_preds, get_affine_transform, transform_preds_with_trans


def get_pred_depth(depth):
  return depth

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan(rot[:, 6] / rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)
  

def ddd_post_process_2d(dets, c, s, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  include_wh = dets.shape[2] > 16
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
    classes = dets[i, :, -1]
    for j in range(opt.num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :3].astype(np.float32),
        get_alpha(dets[i, inds, 3:11])[:, np.newaxis].astype(np.float32),
        get_pred_depth(dets[i, inds, 11:12]).astype(np.float32),
        dets[i, inds, 12:15].astype(np.float32)], axis=1)
      if include_wh:
        top_preds[j + 1] = np.concatenate([
          top_preds[j + 1],
          transform_preds(
            dets[i, inds, 15:17], c[i], s[i], (opt.output_w, opt.output_h))
          .astype(np.float32)], axis=1)
    ret.append(top_preds)
  return ret

def ddd_post_process_3d(dets, calibs):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  for i in range(len(dets)):
    preds = {}
    for cls_ind in dets[i].keys():
      preds[cls_ind] = []
      for j in range(len(dets[i][cls_ind])):
        center = dets[i][cls_ind][j][:2]
        score = dets[i][cls_ind][j][2]
        alpha = dets[i][cls_ind][j][3]
        depth = dets[i][cls_ind][j][4]
        dimensions = dets[i][cls_ind][j][5:8]
        wh = dets[i][cls_ind][j][8:10]
        locations, rotation_y = ddd2locrot(
          center, alpha, dimensions, depth, calibs[0])
        bbox = [center[0] - wh[0] / 2, center[1] - wh[1] / 2,
                center[0] + wh[0] / 2, center[1] + wh[1] / 2]
        pred = [alpha] + bbox + dimensions.tolist() + \
               locations.tolist() + [rotation_y, score]
        preds[cls_ind].append(pred)
      preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)
    ret.append(preds)
  return ret

def ddd_post_process(dets, c, s, calibs, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  dets = ddd_post_process_2d(dets, c, s, opt)
  dets = ddd_post_process_3d(dets, calibs)
  return dets


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret

def generic_post_process(dets, c, s, h, w):
  if not ('scores' in dets):
    return [{}], [{}]
  ret = []

  for i in range(len(dets['scores'])):  # batch 维度
    preds = []
    trans = get_affine_transform(
      c[i], s[i], 0, (w, h), inv=1).astype(np.float32)
    for j in range(len(dets['scores'][i])):   # topk 维度
      item = {}
      item['score'] = dets['scores'][i][j]
      item['ind'] = dets['inds'][i][j]
      item['class'] = int(dets['clses'][i][j]) + 1
      item['ct'] = transform_preds_with_trans(
        (dets['cts'][i][j]).reshape(1, 2), trans).reshape(2)

      # if 'tracking' in dets:
      #   N = dets['tracking'].shape[2] # shape = (batch, K, N, 2)
      #   for k in range(N):
      #     tracking_trans = transform_preds_with_trans(
      #       (dets['tracking'][i][j][k] + dets['cts'][i][j]).reshape(1, 2), trans).reshape(2)
      #     if k == 0:
      #       tracking = (tracking_trans - item['ct'])[np.newaxis, :]  # shape = (1, 2)
      #     else:
      #       tracking_temp = (tracking_trans - item['ct'])[np.newaxis, :]  # shape = (1, 2)
      #       tracking = np.concatenate((tracking, tracking_temp), axis=0)  # shape = (N, 2)
      #   item['tracking'] = tracking

      if 'bboxes' in dets:
        bbox = transform_preds_with_trans(
          dets['bboxes'][i][j].reshape(2, 2), trans).reshape(4)
        item['bbox'] = bbox
      
      preds.append(item)

    ret.append(preds)
  
  return ret


def multi_pose_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  num_classes=24
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
    bbox = bbox.reshape(-1, 4)
    pts = pts.reshape(-1, 34)
    score=dets[i, :, 4]
    score = np.expand_dims(score,axis=1)
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        bbox[ inds, :].astype(np.float32),score[inds],
        pts[inds, :].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret



def multi_pose_post_process_ori(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5], 
       pts.reshape(-1, 34)], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret