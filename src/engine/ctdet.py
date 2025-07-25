from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import numpy as np

from src.loss.losses import FocalLoss, LBHingev2
from src.loss.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from src.utils.decode import ctdet_decode
from src.utils.utils import _sigmoid, _transpose_and_gather_feat
from src.utils.debugger import Debugger
from src.utils.post_process import ctdet_post_process
from src.engine.base_trainer import BaseTrainer



class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.crit = FocalLoss()  # torch.nn.MSELoss()
        self.crit_reg = RegL1Loss()  # RegLoss()
        self.crit_wh = torch.nn.L1Loss(reduction='sum')  # NormRegL1Loss() # RegWeightedL1Loss()
        self.opt = opt
        self.wh_weight = 0.1
        self.hm_weight = 1
        self.off_weight = 1
        self.track_weight = 1
        self.seq_weight = 1
        self.ratios = [1]
        self.num_stacks = 1

    def forward(self, outputs, batches):
        hm_loss, wh_loss, off_loss, track_loss, seq_loss = 0, 0, 0, 0, 0

        for ratio in self.ratios:
            output = outputs[-1][ratio]
            batch = batches[ratio]

            output['hm'] = _sigmoid(output['hm'])
            output['hm_seq'] = _sigmoid(output['hm_seq'])

            hm_loss += self.crit(output['hm'], batch['hm'][:, -1])

            wh_loss += self.crit_reg(
                output['wh'], batch['reg_mask'],
                batch['ind'], batch['wh'])  / self.num_stacks

            off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                    batch['ind'], batch['reg'])  / self.num_stacks

            for i in range(self.opt.seq_len - 1):
                track_loss += self.crit_reg(output['dis'][:, i], batch['dis_mask'][:, i],
                                    batch['dis_ind'][:, i], batch['dis'][:, i]) / (self.opt.seq_len - 1)
            
            for i in range(self.opt.seq_len):
                seq_loss += self.crit(output['hm_seq'][:, i], batch['hm_seq'][:, i]) / self.opt.seq_len

        loss = self.hm_weight * hm_loss + self.wh_weight * wh_loss + \
               self.off_weight * off_loss + self.track_weight * track_loss + self.seq_weight * seq_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'track_loss': track_loss, 'seq_loss': seq_loss}

        return loss, loss_stats


class CtdetTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'track_loss', 'seq_loss']
        # loss_states = ['loss', 'hm_loss']
        loss = CtdetLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=opt.cat_spec_wh, K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        for i in range(1):
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
                                   img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                           dets[i, k, 4], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                           dets_gt[i, k, 4], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]