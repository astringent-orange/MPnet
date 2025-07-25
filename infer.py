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
        # 读取所有帧
        frames = []
        while True:
            ret, img = cap.read()
            if not ret:
                break
            # 确保图像尺寸与训练时一致
            img_input = cv2.resize(img, (opt.img_width, opt.img_height))
            img_input = img_input.astype(np.float32) / 255.
            img_input = (img_input - 0.49965) / 0.08255
            img_input = torch.from_numpy(img_input).permute(2, 0, 1)  # (3, H, W)
            frames.append(img_input)
        cap.release()
        
        if len(frames) == 0:
            print(f'No frames found in {video_name}')
            continue
            
        # 处理时序输入
        seq_len = opt.seq_len
        with open(out_txt, 'w') as f:
            with Bar(f'  {video_name}', max=len(frames)) as frame_bar:
                for i in range(len(frames)):
                    # 构建时序输入：重复当前帧或使用历史帧
                    if i < seq_len - 1:
                        # 前几帧用第一帧填充
                        seq_frames = [frames[0]] * (seq_len - i - 1) + frames[:i+1]
                    else:
                        # 使用历史帧
                        seq_frames = frames[i-seq_len+1:i+1]
                    
                    # 构建时序输入 (B, N, 3, H, W)
                    img_seq = torch.stack(seq_frames, dim=0).unsqueeze(0).to(device)  # (1, seq_len, 3, H, W)
                    
                    # 推理
                    with torch.no_grad():
                        output = model(img_seq, training=False)[1][1]  # 取ret[1]
                    
                    # 解析检测框
                    hm = output['hm'].sigmoid().cpu().numpy()[0, 0]  # (H, W)
                    wh = output['wh'].cpu().numpy()[0]  # (2, H, W)
                    reg = output['reg'].cpu().numpy()[0]  # (2, H, W)
                    
                    thresh = 0.3
                    ys, xs = np.where(hm > thresh)
                    
                    for obj_idx, (y, x) in enumerate(zip(ys, xs)):
                        score = hm[y, x]
                        w, h = wh[0, y, x], wh[1, y, x]
                        # 计算实际坐标
                        x1 = int((x + reg[0, y, x]) * opt.down_ratio)
                        y1 = int((y + reg[1, y, x]) * opt.down_ratio)
                        w = int(w * opt.down_ratio)
                        h = int(h * opt.down_ratio)
                        f.write(f'{frame_idx} {obj_idx} {x1} {y1} {w} {h} 1 -1 -1 -1\n')
                    
                    frame_idx += 1
                    frame_bar.next()
    print(f'Inference done. Results saved to {out_root}') 