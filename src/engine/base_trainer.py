from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from src.utils.data_parallel import DataParallel
from src.utils.utils import AverageMeter
from src.utils.decode import ctdet_decode
from src.utils.post_process import ctdet_post_process
import numpy as np
from torch.cuda.amp import autocast, GradScaler

def post_process(output, meta, num_classes, scale=1):
    # decode
    hm = output['hm'].sigmoid_()
    wh = output['wh']
    reg = output['reg']

    torch.cuda.synchronize()
    dets = ctdet_decode(hm, wh, reg=reg)
    all_dets = []
    for k in dets:
        arr = dets[k].detach().cpu().numpy()
        arr = arr.reshape(-1, arr.shape[-1])
        # 保证每个arr的最后一维为5，不足补零，超出截断
        if arr.shape[-1] < 5:
            pad = np.zeros((arr.shape[0], 5 - arr.shape[-1]), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[-1] > 5:
            arr = arr[:, :5]
        all_dets.append(arr)
    if len(all_dets) > 0:
        all_dets = np.concatenate(all_dets, axis=0)
        all_dets = all_dets[None, ...]  # batch维
    else:
        all_dets = np.zeros((1, 0, 5), dtype=np.float32)
    dets = ctdet_post_process(
        all_dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        if j not in dets[0]:
            dets[0][j] = np.zeros((0, 5), dtype=np.float32)
        else:
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
    return dets

def merge_outputs(detections, num_classes ,max_per_image):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack(
      [results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        # print(batch['input'].shape)
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats


class BaseTrainer(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)
        self.amp_enabled = getattr(opt, 'amp', False)
        self.scaler = GradScaler() if self.amp_enabled else None

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        bar = Bar('Epoch {}/{}'.format(epoch, self.opt.num_epochs), max=num_iters, fill='▇',
                  suffix='%(percent)3d%% | ETA: %(etamsg)7s | loss: %(loss)8.4f | hm: %(hm)8.4f | wh: %(wh)8.4f | off: %(off)8.4f | track: %(track)8.4f | seq: %(seq)8.4f')
        end = time.time()
        for iter_id, (im_id, batch) in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta' and k != 'file_name' and k not in [1, 2, 4]:
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
                elif k in [1, 2, 4]:
                    for item in batch[k]:
                        batch[k][item] = batch[k][item].to(device=opt.device, non_blocking=True)
            if phase == 'train':
                if self.amp_enabled:
                    with autocast():
                        output, loss, loss_stats = model_with_loss(batch)
                    loss = loss.mean()
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output, loss, loss_stats = model_with_loss(batch)
                    loss = loss.mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            else:
                if self.amp_enabled:
                    with autocast():
                        output, loss, loss_stats = model_with_loss(batch)
                    loss = loss.mean()
                else:
                    output, loss, loss_stats = model_with_loss(batch)
                    loss = loss.mean()
            batch_time.update(time.time() - end)

            bar.loss = float(loss.mean().cpu().detach().numpy())
            bar.hm = float(loss_stats['hm_loss'].mean().cpu().detach().numpy())
            bar.wh = float(loss_stats['wh_loss'].mean().cpu().detach().numpy())
            bar.off = float(loss_stats['off_loss'].mean().cpu().detach().numpy())
            bar.track = float(loss_stats['track_loss'].mean().cpu().detach().numpy())
            bar.seq = float(loss_stats['seq_loss'].mean().cpu().detach().numpy())
            # ETA美化为分+秒
            eta = bar.eta
            eta_min = int(eta // 60)
            eta_sec = int(eta % 60)
            bar.etamsg = f'{eta_min}m{eta_sec}s'
            bar.next()

            end = time.time()

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = 1 / 60.

        return ret, results

    def run_eval_epoch(self, phase, epoch, data_loader, dataset):
        model_with_loss = self.model_with_loss

        if len(self.opt.gpus) > 1:
            model_with_loss = self.model_with_loss.module
        model_with_loss.eval()
        torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        # num_iters = 30
        end = time.time()

        def move_to_device(x, device):
            if isinstance(x, torch.Tensor):
                return x.to(device=device, non_blocking=True)
            elif isinstance(x, dict):
                return {k: move_to_device(v, device) for k, v in x.items()}
            elif isinstance(x, list):
                return [move_to_device(v, device) for v in x]
            else:
                return x

        # 新增：验证也用进度条
        bar = Bar('Epoch {}/{} (val)'.format(epoch, self.opt.num_epochs), max=num_iters, fill='▇',
                  suffix='%(percent)3d%% | ETA: %(etamsg)7s | loss: %(loss)8.4f | hm: %(hm)8.4f | wh: %(wh)8.4f | off: %(off)8.4f | track: %(track)8.4f')

        for iter_id, (im_id, batch) in enumerate(data_loader):
            if iter_id >= num_iters:
              break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta' and k != 'file_name':
                    batch[k] = move_to_device(batch[k], opt.device)
            if self.amp_enabled:
                with autocast():
                    output, loss, loss_stats = model_with_loss(batch)
                loss = loss.mean()
            else:
                output, loss, loss_stats = model_with_loss(batch)
                loss = loss.mean()

            inp_height, inp_width = batch['input'].shape[3],batch['input'].shape[4]
            c = np.array([inp_width / 2., inp_height / 2.], dtype=np.float32)
            s = max(inp_height, inp_width) * 1.0

            meta = {'c': c, 's': s,
                    'out_height': inp_height,
                    'out_width': inp_width}

            # 修正output结构，确保有'hm' key
            def find_hm_dict(d):
                if isinstance(d, dict):
                    if 'hm' in d:
                        return d
                    for v in d.values():
                        res = find_hm_dict(v)
                        if res is not None:
                            return res
                elif isinstance(d, (list, tuple)):
                    for v in d:
                        res = find_hm_dict(v)
                        if res is not None:
                            return res
                return None
            # 兼容list/tuple输出
            out_for_post = output
            if isinstance(out_for_post, (list, tuple)):
                out_for_post = out_for_post[0]
            if not (isinstance(out_for_post, dict) and 'hm' in out_for_post):
                out_for_post = find_hm_dict(out_for_post)
            if not (isinstance(out_for_post, dict) and 'hm' in out_for_post):
                print('[ERROR] output for post_process does not contain key "hm". keys:', out_for_post.keys() if isinstance(out_for_post, dict) else type(out_for_post))
                raise KeyError('hm')

            dets = post_process(out_for_post, meta, opt.num_classes)
            ret = merge_outputs([dets[0]], opt.num_classes, max_per_image=opt.max_objs)
            results[im_id.numpy().astype(np.int32)[0]] = ret

            loss = loss.mean()
            batch_time.update(time.time() - end)

            # 进度条显示
            bar.loss = float(loss.mean().cpu().detach().numpy())
            bar.hm = float(loss_stats['hm_loss'].mean().cpu().detach().numpy())
            bar.wh = float(loss_stats['wh_loss'].mean().cpu().detach().numpy())
            bar.off = float(loss_stats['off_loss'].mean().cpu().detach().numpy())
            bar.track = float(loss_stats['track_loss'].mean().cpu().detach().numpy())
            # ETA美化为分+秒
            eta = bar.eta
            eta_min = int(eta // 60)
            eta_sec = int(eta % 60)
            bar.etamsg = f'{eta_min}m{eta_sec}s'
            bar.next()

            end = time.time()

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        # coco_evaluator.accumulate()
        # coco_evaluator.summarize()
        stats1, _ = dataset.run_eval(results, opt.save_dir, 'latest')
        ret['time'] = 1 / 60.
        # 兼容不同类型的stats1
        if isinstance(stats1, dict) and 'ap50' in stats1:
            ret['ap50'] = stats1['ap50']
        elif isinstance(stats1, (list, tuple)) and len(stats1) > 1:
            ret['ap50'] = stats1[1]
        elif isinstance(stats1, dict) and 'precision' in stats1:
            ret['ap50'] = stats1['precision']
        else:
            ret['ap50'] = 0.0

        print('Eval summary: map50={:.4f}, precision={:.4f}, recall={:.4f}, MOTA={:.4f}, IDF1={:.4f}'.format(
                stats1.get('precision', 0.0) if isinstance(stats1, dict) else 0.0,
                stats1.get('precision', 0.0) if isinstance(stats1, dict) else 0.0,
                stats1.get('recall', 0.0) if isinstance(stats1, dict) else 0.0,
                stats1.get('MOTA', 0.0) if isinstance(stats1, dict) else 0.0,
                stats1.get('IDF1', 0.0) if isinstance(stats1, dict) else 0.0))

        # 统计耗时
        total_time = time.time() - end
        total_min = int(total_time // 60)
        total_sec = int(total_time % 60)
        val_time_str = 'Val time: {}m{}s\n'.format(total_min, total_sec)

        # 只追加一行进度条统计、一行Eval summary和一行val耗时到eval_results.txt
        import datetime, os
        bar_str = 'Epoch {}/{} (val) | loss: {:8.4f} | hm: {:8.4f} | wh: {:8.4f} | off: {:8.4f} | track: {:8.4f}\n'.format(
            epoch, self.opt.num_epochs,
            ret.get('loss', 0.0), ret.get('hm_loss', 0.0), ret.get('wh_loss', 0.0), ret.get('off_loss', 0.0), ret.get('track_loss', 0.0))
        summary_str = 'Eval summary: map50={:.4f}, precision={:.4f}, recall={:.4f}, MOTA={:.4f}, IDF1={:.4f}\n'.format(
            stats1.get('map50', 0.0) if isinstance(stats1, dict) else 0.0,
            stats1.get('precision', 0.0) if isinstance(stats1, dict) else 0.0,
            stats1.get('recall', 0.0) if isinstance(stats1, dict) else 0.0,
            stats1.get('MOTA', 0.0) if isinstance(stats1, dict) else 0.0,
            stats1.get('IDF1', 0.0) if isinstance(stats1, dict) else 0.0)
        with open(os.path.join(opt.save_dir, 'eval_results.txt'), 'a') as f:
            f.write(bar_str)
            f.write(summary_str)
            f.write(val_time_str)

        return ret, results, stats1

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader, dataset):
        # 执行一个 epoch 的验证，返回验证日志
        return self.run_eval_epoch('val', epoch, data_loader, dataset)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)