from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import torch

class Logger(object):
    def __init__(self, opt):
        """
        日志记录器，用于将训练过程中的关键信息输出到指定目录下的文件。

        主要输出文件说明：
        1. 训练参数(opt)和环境信息会被写入到 {opt.save_dir}/opt.txt
        2. 日志内容会被写入到 {opt.save_dir}/log.txt

        其中 opt.save_dir 由命令行参数或配置文件指定，通常为项目下的某个保存目录。
        """

        # 收集所有opt参数，写入opt.txt
        args = dict((name, getattr(opt, name)) for name in dir(opt)
                    if not name.startswith('_'))
        file_name = os.path.join(opt.save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> torch version: {}\n'.format(torch.__version__))
            opt_file.write('==> cudnn version: {}\n'.format(
                torch.backends.cudnn.version()))
            opt_file.write('==> Cmd:\n')
            opt_file.write(str(sys.argv))
            opt_file.write('\n==> Opt:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))

        # 打开日志文件，后续所有日志内容会写入该文件
        self.log = open(opt.save_dir + '/log.txt', 'w')
        self.start_line = True

    def write(self, txt):
        """
        写入日志内容到 log.txt 文件。
        每条新日志会自动加上时间戳。
        """
        if self.start_line:
            time_str = time.strftime('%Y-%m-%d %H:%M:%S')
            self.log.write('{}: {}'.format(time_str, txt))
        else:
            self.log.write(txt)
        self.start_line = False
        if '\n' in txt:
            self.start_line = True
            self.log.flush()
    
    def close(self):
        """
        关闭日志文件。
        """
        self.log.close()
