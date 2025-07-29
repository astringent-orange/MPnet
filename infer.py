'''
推理代码

输入：
    - data/dataset/test/ 目录下的所有 .avi 视频文件。
    - 模型权重由 opt.model_path 指定。

输出：
    - 每个视频在 data/outputs/时间戳/ 目录下生成一个同名 .txt 文件。
    - 每个 txt 文件内容为该视频所有帧的检测结果。
    - 每行格式：帧号 物体id x y w h 1 -1 -1 -1
      其中：
        - 帧号：当前帧的索引（从0开始）
        - 物体id：该帧内检测框的序号
        - x, y：检测框左上角坐标（整数）
        - w, h：检测框宽度和高度（整数）
        - 1：目标类别，固定为1
        - -1 -1 -1：占位，保持与数据集label格式一致

流程：
    - 遍历所有avi视频，逐帧读取并推理。
    - 对每帧，提取热力图中心点作为目标，输出检测框。
    - 结果保存为txt，便于后续评测或可视化。
'''

import os
os.environ['TORCH_HOME'] = '/data/hdd3/zhangmian/Tmp/torch/'
import time
import torch
import numpy as np
from src.utils.opts import opts
from src.models.model import Model
from src.models.utils import load_model
from datetime import datetime
import cv2
from glob import glob
from src.utils.tools import print_banner
from progress.bar import Bar
from src.utils.discheck import check

if __name__ == '__main__':
    opt = opts().parse()
    
    # 设置默认图像尺寸，与训练时保持一致
    if not hasattr(opt, 'img_height') or not hasattr(opt, 'img_width'):
        opt.img_height = 512
        opt.img_width = 512
        print(f'Using default image size: {opt.img_width}x{opt.img_height}')
    
    # 设置device，与train一致
    main_gpu = opt.gpus[0] if hasattr(opt, 'gpus') else 0
    device = torch.device(f'cuda:{main_gpu}' if torch.cuda.is_available() and main_gpu >= 0 else 'cpu')
    print(f'Using device: {device}')
    
    # 加载模型
    model = Model(opt)
    model_path = opt.model_path
    assert os.path.exists(model_path), f'No model found at {model_path}'
    model = load_model(model, model_path)
    model.to(device)
    model.eval()

    # 输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root = os.path.join('data/outputs', timestamp)
    os.makedirs(out_root, exist_ok=True)

    print_banner('Start inference')

    # 遍历所有avi视频
    video_dir = os.path.join(opt.dataroot, 'test')
    video_list = sorted(glob(os.path.join(video_dir, '*.avi')))
    print(f'Found {len(video_list)} videos')
    
    for video_path in video_list:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_txt = os.path.join(out_root, f'{video_name}.txt')
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        frames = []
        while True:
            ret, img = cap.read()
            if not ret:
                break
            img_input = cv2.resize(img, (opt.img_width, opt.img_height))
            img_input = img_input.astype(np.float32) / 255.
            img_input = (img_input - 0.49965) / 0.08255
            img_input = torch.from_numpy(img_input).permute(2, 0, 1)
            frames.append(img_input)
        cap.release()
        if len(frames) == 0:
            print(f'No frames found in {video_name}')
            continue
        seq_len = opt.seq_len
        dets_buffer = []
        inds_buffer = []
        dis_buffer = []
        file_folder_buffer = []
        with open(out_txt, 'w') as f:
            with Bar(f'  {video_name}', max=len(frames)) as frame_bar:
                for i in range(len(frames)):
                    if i < seq_len - 1:
                        seq_frames = [frames[0]] * (seq_len - i - 1) + frames[:i+1]
                    else:
                        seq_frames = frames[i-seq_len+1:i+1]
                    img_seq = torch.stack(seq_frames, dim=0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(img_seq, training=False)[1][1]
                    hm = output['hm'].sigmoid().cpu().numpy()[0, 0]
                    wh = output['wh'].cpu().numpy()[0]
                    reg = output['reg'].cpu().numpy()[0]
                    dis = output['dis'].cpu().numpy()[0]  # (seq_len-1, 2, H, W)
                    thresh = 0.2
                    ys, xs = np.where(hm > thresh)
                    dets = []
                    inds = []
                    for obj_idx, (y, x) in enumerate(zip(ys, xs)):
                        score = hm[y, x]
                        w, h = wh[0, y, x], wh[1, y, x]
                        x1 = int((x + reg[0, y, x]) * opt.down_ratio)
                        y1 = int((y + reg[1, y, x]) * opt.down_ratio)
                        w = int(w * opt.down_ratio)
                        h = int(h * opt.down_ratio)
                        x2 = x1 + w
                        y2 = y1 + h
                        dets.append([x1, y1, x2, y2, score, 1, obj_idx])
                        inds.append(y * hm.shape[1] + x)
                    dets_buffer.append(dets)
                    inds_buffer.append(np.array(inds, dtype=np.int64))
                    dis_buffer.append(dis)
                    file_folder_buffer.append(video_name)
                    # 每seq_len帧做一次优化
                    if (i + 1) % seq_len == 0 or i == len(frames) - 1:
                        dets_seq = dets_buffer[-seq_len:]
                        dis_seq = dis_buffer[-seq_len:]
                        inds_seq = inds_buffer[-seq_len:]
                        file_seq = file_folder_buffer[-seq_len:]
                        dets_seq_new, inds_seq_new = check(opt, file_seq, dets_seq, dis_seq, inds_seq)
                        dets_buffer[-seq_len:] = dets_seq_new
                        inds_buffer[-seq_len:] = inds_seq_new
                    frame_bar.next()
            # 写入优化后的检测结果
            for frame_idx, dets in enumerate(dets_buffer):
                for obj in dets:
                    x1, y1, x2, y2, score, cls, ind = obj
                    w = x2 - x1
                    h = y2 - y1
                    f.write(f'{frame_idx} {ind} {int(x1)} {int(y1)} {int(w)} {int(h)} 1 -1 -1 -1\n')

    print(f'Inference done. Results saved to {out_root}') 