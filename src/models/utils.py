import torch
from thop import profile

def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    # 兼容不同保存格式
    if 'state_dict' in checkpoint:
        state_dict_ = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict_ = checkpoint['model_state_dict']
    else:
        state_dict_ = checkpoint
    model.load_state_dict(state_dict_, strict=False)
    print(f'loaded {model_path}')
    return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    heads = {'hm': 1, 'wh': 2, 'reg': 2, 'dis': 2}
    model_nameAll = ['DLADCN', 'ResFPN']
    device = 'cuda:0'

    out = {}

    for model_name in model_nameAll:
        net = get_det_net(heads, model_name).to(device)
        input = torch.rand(1,1,3,512,512).to(device)
        flops, params = profile(net, inputs=(input, False, 'flag'))
        out[model_name] = [flops, params]

    for k,v in out.items():
        print('---------------------------------------------')
        print(k + '   Number of flops: %.2fG' % (v[0] / 1e9))
        print(k + '   Number of params: %.2fM' % (v[1] / 1e6))