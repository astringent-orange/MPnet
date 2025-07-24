'''
训练模型

断点续训说明：
- 训练前会优先检查opt.model_path（如有）是否存在。
- 若存在，则自动加载模型和优化器参数，并从上次中断的epoch+1继续训练。
- 若不存在，则回退到save_dir/model_last.pth。
- 若都不存在，则从头开始训练。
- 每个epoch后会自动保存最新断点到model_last.pth。
- 支持意外中断后无缝恢复训练。
'''

import os
os.environ['TORCH_HOME'] = '/data/hdd3/zhangmian/Tmp/torch/'
from src.utils.opts import opts
from src.utils.tools import print_banner

import torch
from torch.utils.data import DataLoader

# 导入数据集和模型
from src.data.datasets import MPDataset
from src.models.model import Model 
from src.models.utils import load_model, save_model

from src.engine.ctdet import CtdetTrainer

from src.utils.logger import Logger

import time


def train(opt):
    # ********************** 准备训练 **********************
    print_banner('Preparing training')

    # 设置初始化参数
    torch.manual_seed(opt.seed)
    main_gpu = opt.gpus[0]
    opt.device = torch.device(f'cuda:{main_gpu}' if main_gpu >= 0 else 'cpu')
    val_intervals = opt.val_intervals
    print(f'Using device: {opt.device}')

    # 创建数据集
    train_dataset = MPDataset(opt, dataset_type='train')
    val_dataset = MPDataset(opt, dataset_type='val')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    # 创建模型
    model = Model(opt)
    model.to(opt.device)
    print('Create model:', model.__class__.__name__)

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    print('Create optimizer:', optimizer.__class__.__name__)

    # 断点续训：优先加载opt.model_path
    resume_path = getattr(opt, 'model_path', None)
    if resume_path is None or not os.path.exists(resume_path):
        resume_path = os.path.join(opt.save_dir, 'model_last.pth')
    start_epoch = 0
    best = -1   # 最佳验证集 ap50
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=opt.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best = checkpoint.get('best', -1)
            print(f'Resume training from epoch {start_epoch+1} (loaded from {resume_path})')
        else:
            # 只保存了模型参数
            model.load_state_dict(checkpoint)
            print(f'Loaded model weights from {resume_path}, start training from scratch.')
            start_epoch = 0
            best = -1
    else:
        print('Start training from scratch.')

    # 创建训练器
    trainer = CtdetTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.device)
    print('Create trainer:', trainer.__class__.__name__)

    # 创建保存目录
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # 创建记录器
    logger = Logger(opt)
    print('Create logger:', logger.__class__.__name__)


    # ********************** 开始训练 **********************
    print_banner('Start training')
    elapsed_times = []
    total_start_time = time.time()
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        start_time = time.time()
        # 执行一个 epoch 的训练，返回训练日志
        log_dict_train, _ = trainer.train(epoch, train_loader)

        # 记录当前 epoch
        logger.write('epoch: {} |'.format(epoch))

        # 保存最新模型参数和断点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best': best
        }, os.path.join(opt.save_dir, 'model_last.pth'))

        # 打印训练日志信息
        for k, v in log_dict_train.items():
            logger.write('{} {:8f} | '.format(k, v))

        # 判断是否需要进行验证（根据 val_intervals）
        if val_intervals > 0 and epoch % val_intervals == 0:
            # 在验证集上评估模型
            with torch.no_grad():
                log_dict_val, preds, stats = trainer.val(epoch, val_loader, val_dataset)
            # 打印验证日志信息
            for k, v in log_dict_val.items():
                logger.write('{} {:8f} | '.format(k, v))
            logger.write('eval results: ')
            # 打印评估指标
            for k, v in stats.items():
                logger.write('{}: {:8f} | '.format(k, v))
            # 如果当前 ap50 超过历史最佳，则保存为最佳模型
            if log_dict_val['ap50'] > best:
                best = log_dict_val['ap50']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best': best
                }, os.path.join(opt.save_dir, 'model_best.pth'))
        logger.write('\n')

        # 统计并输出本epoch耗时
        elapsed_time = time.time() - start_time
        elapsed_times.append(elapsed_time)
        epoch_min = int(elapsed_time // 60)
        epoch_sec = int(elapsed_time % 60)
        print('Epoch {} time: {}m{}s'.format(epoch, epoch_min, epoch_sec))
        logger.write('Epoch {} time: {}m{}s\n'.format(epoch, epoch_min, epoch_sec))

        # 按照 lr_step 调整学习率
        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    # ********************** 结束训练 **********************
    total_time = time.time() - total_start_time
    avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
    total_min = int(total_time // 60)
    total_sec = int(total_time % 60)
    avg_min = int(avg_time // 60)
    avg_sec = int(avg_time % 60)
    print('Total training time: {}m{}s, Average per epoch: {}m{}s'.format(total_min, total_sec, avg_min, avg_sec))
    logger.write('Total training time: {}m{}s, Average per epoch: {}m{}s\n'.format(total_min, total_sec, avg_min, avg_sec))
    print_banner('Training finished')
    logger.close()


if __name__ == "__main__":
    opt = opts().parse()
    train(opt)