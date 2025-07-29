# MP2net 

[![Python](https://img.shields.io/badge/python-3.9+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

## 项目简介 ✨

本项目使用东方慧眼杯比赛提供的数据集来训练**MP2Net**

## 主要功能 🛠️
- 🎯 **多目标检测**：基于DLA主干网络，支持高分辨率遥感图像的目标检测。
- 🔄 **时序掩码传播**：利用时序信息提升检测与跟踪的鲁棒性。
- 🏃 **多目标跟踪（MOT）**：支持MOTA、IDF1等主流跟踪评测指标。
- 💾 **断点续训**：支持训练中断后自动恢复。
- 📹 **推理与结果导出**：支持对avi视频批量推理，输出与数据集一致的txt格式。

## 文件结构 🗂️
```
MPnet/
├── train.py              # 训练主脚本
├── infer.py              # 推理脚本（支持avi视频）
├── test_eval.py          # 测试验证脚本
├── test_dataset.py       # 测试数据集脚本
├── src/
│   ├── models/           # 模型结构、权重加载等
│   ├── data/             # 数据集定义与处理
│   ├── engine/           # 训练与评测核心逻辑
│   ├── utils/            # 工具函数、日志、参数解析等
├── data/
│   ├── dataset/          # 数据集目录（images/、labels/、test/）
│   └── outputs/          # 推理结果输出目录
├── experiments/          # 训练日志、模型权重、评测结果等
└── README.md             # 项目说明
```

## 依赖环境 ⚙️
- Python 3.9
- PyTorch >= 2.1.0
- OpenCV
- numpy
- mmcv (含DeformConv)
- 其他依赖见requirements.txt

## 环境准备 🧰

建议使用虚拟环境（如conda）以避免依赖冲突。

1. 创建并激活虚拟环境（以conda为例）：
   ```bash
   conda create -n mp2net python=3.9 -y
   conda activate mp2net
   ```
2. 安装依赖：
   ```bash
   pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html   # 建议最先安装
   pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

## 数据集准备 📦

数据集下载：[百度网盘](https://pan.baidu.com/s/10MHXqTr2JlVwrSi3LDcvIw?pwd=dsey) 提取码: dsey

将下载的数据集dataset放置于data目录下，最终结构如下
```
data/
└── dataset/
    ├── train/                  # 训练集
    │   ├── images/             # 训练图片
    │   │   ├── 1-2_000000.jpg
    │   │   ├── 1-2_000001.jpg
    │   │   └── ... 
    │   └── labels/             # 训练标签
    │       ├── 1-2_000000.txt
    │       ├── 1-2_000001.txt
    │       └── ...
    ├── val/                    # 验证集
    │   ├── images/             # 验证图片
    │   └── labels/             # 验证标签
    └── test/                   # 测试集
        └── *.avi               # 测试视频（avi格式）
  
```
其中images为视频的单帧图像，而labels为每帧图像所含物体信息，以1-2_000000.txt举例，格式如下
```
帧号  物体id  检测框左上角x  检测框左上角y  检测框宽w  检测框高h  类别（固定为1）  
0     0       807           367           6         5         1         -1   -1   -1
0     1       962           755           8         9         1         -1   -1   -1
```



---

## 如何运行 ▶️

### 1. 训练模型
```bash
python train.py --gpus '0,1' --batch_size 2 --num_epochs 20 --val_intervals 2 
```
- 支持断点续训，无需特殊参数，可用`--model_path`指定权重文件恢复。

### 2. 推理（对avi视频）
```bash
python infer.py --gpus '0' --model_path ./experiments/xxx/model_best.pth --dataroot ./data/dataset
```
- 输出结果保存在`data/outputs/时间戳/`下，每个视频一个同名txt。

### 3. 测试部分代码能否运行
```bash
python test_dataset.py
python test_eval.py 
```
- 随机生成预测结果，测试评测流程和指标统计。

## 参数设置说明 📝
- 所有参数均可通过命令行传递或在`src/utils/opts.py`中设置默认值。
- 常用参数：
  - `--gpus`：使用的GPU编号，如`'0'`或`'0,1'`
  - `--batch_size`：训练batch size
  - `--num_epochs`：训练总轮数
  - `--val_intervals`：每多少个epoch验证一次
  - `--save_dir`：实验输出目录
  - `--model_path`：恢复训练或推理时加载的权重文件
  - `--dataroot`：数据集根目录
  - `--seq_len`：时序长度
  - `--max_objs`：每帧最大目标数
- 详细参数见`src/utils/opts.py`。

## 训练/推理/评测结果 📊
- 训练日志、模型权重、评测结果等保存在`experiments/`下的对应实验目录。
- 推理结果（txt）保存在`data/outputs/时间戳/`下。
- 评测结果（如mAP50、MOTA、IDF1等）会自动写入`eval_results.txt`。

---

## 参考
```
@article{zhao2024mp2net,
  author={Zhao, Manqi and Li, Shengyang and Wang, Han and Yang, Jian and Sun, Yuhan and Gu, Yanfeng},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={MP2Net: Mask Propagation and Motion Prediction Network for Multiobject Tracking in Satellite Videos}, 
  year={2024},
  volume={62},
  pages={1-15},
  publisher={IEEE}
}
```
---

如有问题或建议，欢迎提issue或联系作者。
