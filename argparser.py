import json
from collections import OrderedDict
import torch
import sys
import os
import datetime
from utils import str2bool
import argparse

try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False
try:
    import apex
    is_apex_available = True
except Exception:
    is_apex_available = False

def _args2config(args, keys, json_keys):
    if json_keys is None:
        json_keys = []

    args = vars(args)

    config = OrderedDict()
    for key in keys:
        value = args[key]
        if value is None:
            continue

        if key in json_keys and isinstance(value, str):
            value = json.loads(value)

        config[key] = value

    return config


def _get_model_config(args):
    keys = [
        'arch',
        'input_shape',
        'n_classes',
        # vgg
        'n_channels',
        'n_layers',
        'use_bn',
        #
        'base_channels',
        'block_type',
        'depth',
        # resnet_preact, se_resnet_preact
        'remove_first_relu',
        'add_last_bn',
        'preact_stage',
        # wrn
        'widening_factor',
        # densenet
        'growth_rate',
        'compression_rate',
        # wrn, densenet
        'drop_rate',
        # pyramidnet
        'pyramid_alpha',
        # resnext
        'cardinality',
        # shake_shake
        'shake_forward',
        'shake_backward',
        'shake_image',
        # se_resnet_preact
        'se_reduction',
    ]
    json_keys = ['preact_stage']
    config = _args2config(args, keys, json_keys)
    return config


def _check_optim_config(config):
    optimizer = config['optimizer']
    for key in ['base_lr', 'weight_decay']:
        message = 'Key `{}` must be specified.'.format(key)
        assert key in config.keys(), message
    if optimizer == 'sgd':
        for key in ['momentum', 'nesterov']:
            message = 'When using SGD, key `{}` must be specified.'.format(key)
            assert key in config.keys(), message
    elif optimizer == 'adam':
        for key in ['betas']:
            message = 'When using Adam, key `{}` must be specified.'.format(
                key)
            assert key in config.keys(), message
    elif optimizer == 'lars':
        for key in ['momentum']:
            message = 'When using LARS, key `{}` must be specified.'.format(
                key)
            assert key in config.keys(), message

    scheduler = config['scheduler']
    if scheduler == 'multistep':
        for key in ['milestones', 'lr_decay']:
            message = 'Key `{}` must be specified.'.format(key)
            assert key in config.keys(), message
    elif scheduler == 'cosine':
        for key in ['lr_min']:
            message = 'Key `{}` must be specified.'.format(key)
            assert key in config.keys(), message
    elif scheduler == 'sgdr':
        for key in ['lr_min', 'T0', 'Tmult']:
            message = 'Key `{}` must be specified.'.format(key)
            assert key in config.keys(), message


def _get_optim_config(args):
    keys = [
        'epochs',
        'batch_size',
        'ghost_batch_size',
        'optimizer',
        'base_lr',
        'weight_decay',
        'no_weight_decay_on_bn',
        'momentum',
        'nesterov',
        'gradient_clip',
        'scheduler',
        'milestones',
        'lr_decay',
        'lr_min',
        'T0',
        'Tmult',
        'betas',
        'lars_eps',
        'lars_thresh',
    ]
    json_keys = ['milestones', 'betas']
    config = _args2config(args, keys, json_keys)

    _check_optim_config(config)

    return config


def _get_data_config(args):
    keys = [
        'dataset',
        'n_classes',
        'num_workers',
        'batch_size',
        'use_random_crop',
        'random_crop_padding',
        'use_horizontal_flip',
        'use_cutout',
        'use_dual_cutout',
        'cutout_size',
        'cutout_prob',
        'cutout_inside',
        'use_random_erasing',
        'dual_cutout_alpha',
        'random_erasing_prob',
        'random_erasing_area_ratio_range',
        'random_erasing_min_aspect_ratio',
        'random_erasing_max_attempt',
        'use_mixup',
        'mixup_alpha',
        'use_ricap',
        'ricap_beta',
        'use_label_smoothing',
        'label_smoothing_epsilon',
        'use_cl_lp',
        'lp_alpha',
        'lp_p',
        'is_select',
        'start_epoch',
        'end_epoch',
        'use_rgl_cl_lp',
        'rgl_type',
        'rgl_interval',
    ]
    json_keys = ['random_erasing_area_ratio_range']
    config = _args2config(args, keys, json_keys)
    config['use_gpu'] = args.device != 'cpu'
    _check_data_config(config)
    return config


def _check_data_config(config):
    if config['use_cutout'] and config['use_dual_cutout']:
        raise ValueError(
            'Only one of `use_cutout` and `use_dual_cutout` can be `True`.')
    if sum([
            config['use_mixup'], config['use_ricap'], config['use_dual_cutout']
    ]) > 1:
        raise ValueError(
            'Only one of `use_mixup`, `use_ricap` and `use_dual_cutout` can be `True`.'
        )


def _get_run_config(args):
    keys = [
        'outdir',
        'seed',
        'test_first',
        'device',
        'fp16',
        'use_amp',
        'tensorboard',
        'tensorboard_train_images',
        'tensorboard_test_images',
        'tensorboard_model_params',
        'is_final'
    ]
    config = _args2config(args, keys, None)

    return config


def _get_env_info(args):
    info = OrderedDict({
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
    })

    def _get_device_info(device_id):
        name = torch.cuda.get_device_name(device_id)
        capability = torch.cuda.get_device_capability(device_id)
        capability = '{}.{}'.format(*capability)
        return name, capability

    if args.device != 'cpu':
        for gpu_id in range(torch.cuda.device_count()):
            name, capability = _get_device_info(gpu_id)
            info['gpu{}'.format(gpu_id)] = OrderedDict({
                'name':
                name,
                'capability':
                capability,
            })

    return info


def _cleanup_args(args):
    # architecture
    if args.arch == 'vgg':
        args.base_channels = None
        args.depth = None
    if args.arch != 'vgg':
        args.n_channels = None
        args.n_layers = None
        args.use_bn = None
    if args.arch not in [
            'resnet', 'resnet_preact', 'densenet', 'pyramidnet',
            'se_resnet_preact'
    ]:
        args.block_type = None
    if args.arch not in ['resnet_preact', 'se_resnet_preact']:
        args.remove_first_relu = None
        args.add_last_bn = None
        args.preact_stage = None
    if args.arch != 'wrn':
        args.widening_factor = None
    if args.arch != 'densenet':
        args.growth_rate = None
        args.compression_rate = None
    if args.arch not in ['wrn', 'densenet']:
        args.drop_rate = None
    if args.arch != 'pyramidnet':
        args.pyramid_alpha = None
    if args.arch != 'resnext':
        args.cardinality = None
    if args.arch != 'shake_shake':
        args.shake_forward = None
        args.shake_backward = None
        args.shake_image = None
    if args.arch != 'se_resnet_preact':
        args.se_reduction = None

    # optimizer
    if args.optimizer not in ['sgd', 'lars']:
        args.momentum = None
    if args.optimizer != 'sgd':
        args.nesterov = None
    if args.optimizer != 'adam':
        args.betas = None
    if args.optimizer != 'lars':
        args.lars_eps = None
        args.lars_thresh = None

    # scheduler
    if args.scheduler != 'multistep':
        args.milestones = None
        args.lr_decay = None
    if args.scheduler not in ['cosine', 'sgdr']:
        args.lr_min = None
    if args.scheduler != 'sgdr':
        args.T0 = None
        args.Tmult = None

    # standard data augmentation
    if args.use_random_crop is None:
        if args.dataset in ['CIFAR10', 'CIFAR100', 'FashionMNIST', 'KMNIST']:
            args.use_random_crop = True
        else:
            args.use_random_crop = False
    if not args.use_random_crop:
        args.random_crop_padding = None
    if args.use_horizontal_flip is None:
        if args.dataset in ['CIFAR10', 'CIFAR100', 'FashionMNIST']:
            args.use_horizontal_flip = True
        else:
            args.use_horizontal_flip = False

    # (dual-)cutout
    if not args.use_cutout and not args.use_dual_cutout:
        args.cutout_size = None
        args.cutout_prob = None
        args.cutout_inside = None
    if not args.use_dual_cutout:
        args.dual_cutout_alpha = None

    # random erasing
    if not args.use_random_erasing:
        args.random_erasing_prob = None
        args.random_erasing_area_ratio_range = None
        args.random_erasing_min_aspect_ratio = None
        args.random_erasing_max_attempt = None

    # mixup
    if not args.use_mixup:
        args.mixup_alpha = None

    # RICAP
    if not args.use_ricap:
        args.ricap_beta = None

    # label smoothing
    if not args.use_label_smoothing:
        args.label_smoothing_epsilon = None

    # TensorBoard
    if not args.tensorboard:
        args.tensorboard_train_images = False
        args.tensorboard_test_images = False
        args.tensorboard_model_params = False

    # data
    if args.dataset == 'CIFAR10':
        args.input_shape = (1, 3, 32, 32)
        args.n_classes = 10
    elif args.dataset == 'CIFAR100':
        args.input_shape = (1, 3, 32, 32)
        args.n_classes = 100
    elif 'MNIST' in args.dataset:
        args.input_shape = (1, 1, 28, 28)
        args.n_classes = 10

    return args


def _set_default_values(args):
    if args.config is not None:
        with open(args.config, 'r') as fin:
            config = json.load(fin)

        d_args = vars(args)
        for config_key, default_config in config.items():
            if config_key == 'env_info':
                continue

            for key, default_value in default_config.items():
                if key not in d_args.keys() or d_args[key] is None:
                    setattr(args, key, default_value)

    return args


def get_config(args):
    if args.arch is None and args.config is None:
        raise RuntimeError(
            'One of args.arch and args.config must be specified')
    if args.config is None:
        args.config = 'configs/{}.json'.format(args.arch)

    args = _set_default_values(args)
    args = _cleanup_args(args)
    config = OrderedDict({
        'model_config': _get_model_config(args),
        'optim_config': _get_optim_config(args),
        'data_config': _get_data_config(args),
        'run_config': _get_run_config(args),
        'env_info': _get_env_info(args),
    })

    dir = get_dir(config,sys.argv)

    #print(dir)
    dir = dir+'-T'
    index = 1
    while(True):
        if not os.path.isdir(dir+str(index)):
            config['run_config']['outdir']=dir+str(index)
            break
        else:
            index +=1

    return config


def get_dir(config,argv):
    dir = 'result/' + config['data_config']['dataset'] + '/' + config['model_config']['arch'] + '/' \
          + config['model_config']['arch']

    if config['run_config']['is_final']:
        dir ='final_'+dir

    parm = (' '.join(argv))
    parm = parm.replace('--', '-')
    arg = parm.split('-')

    configkey = list(config['model_config'].keys())+list(config['optim_config'].keys()) \
                +list(config['data_config'].keys())+list(config['run_config'].keys())

    for setting in arg:
        key = setting.split(' ')[0]

        if key in configkey and key not in['dataset','arch','outdir']:
            dir = dir + '-' + key
            rest = setting.replace(key, '').replace(' ', '')
            if rest is not '':
                dir = dir + '-' + rest

    dir.replace('\r','').replace('\n','')

    return dir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str)
    parser.add_argument('--config', type=str)

    # model config (VGG)
    parser.add_argument('--n_channels', type=str)
    parser.add_argument('--n_layers', type=str)
    parser.add_argument('--use_bn', type=str2bool)
    #
    parser.add_argument('--base_channels', type=int)
    parser.add_argument('--block_type', type=str)
    parser.add_argument('--depth', type=int)
    # model config (ResNet-preact)
    parser.add_argument('--remove_first_relu', type=str2bool)
    parser.add_argument('--add_last_bn', type=str2bool)
    parser.add_argument('--preact_stage', type=str)
    # model config (WRN)
    parser.add_argument('--widening_factor', type=int)
    # model config (DenseNet)
    parser.add_argument('--growth_rate', type=int)
    parser.add_argument('--compression_rate', type=float)
    # model config (WRN, DenseNet)
    parser.add_argument('--drop_rate', type=float)
    # model config (PyramidNet)
    parser.add_argument('--pyramid_alpha', type=int)
    # model config (ResNeXt)
    parser.add_argument('--cardinality', type=int)
    # model config (shake-shake)
    parser.add_argument('--shake_forward', type=str2bool)
    parser.add_argument('--shake_backward', type=str2bool)
    parser.add_argument('--shake_image', type=str2bool)
    # model config (SENet)
    parser.add_argument('--se_reduction', type=int)

    parser.add_argument('--outdir', type=str)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--test_first', type=str2bool, default=True)
    parser.add_argument('--device', type=str, default='cuda')

    # TensorBoard configuration
    parser.add_argument(
        '--tensorboard', dest='tensorboard', action='store_true', default=True)
    parser.add_argument(
        '--no-tensorboard', dest='tensorboard', action='store_false')
    parser.add_argument('--tensorboard_train_images', action='store_true')
    parser.add_argument('--tensorboard_test_images', action='store_true')
    parser.add_argument('--tensorboard_model_params', action='store_true')
    parser.add_argument('--is_final', action='store_true')



    # configuration of optimizer
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--ghost_batch_size', type=int)
    parser.add_argument(
        '--optimizer', type=str, choices=['sgd', 'adam', 'lars'])
    parser.add_argument('--gradient_clip', type=float)
    parser.add_argument('--base_lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--no_weight_decay_on_bn', action='store_true')
    # configuration for SGD
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--nesterov', type=str2bool)
    # configuration for learning rate scheduler
    parser.add_argument(
        '--scheduler',
        type=str,
        choices=['none', 'multistep', 'cosine', 'sgdr'])
    # configuration for multi-step scheduler]
    parser.add_argument('--milestones', type=str)
    parser.add_argument('--lr_decay', type=float)
    # configuration for cosine-annealing scheduler and SGDR scheduler
    parser.add_argument('--lr_min', type=float, default=0)
    # configuration for SGDR scheduler
    parser.add_argument('--T0', type=int)
    parser.add_argument('--Tmult', type=int)
    # configuration for Adam
    parser.add_argument('--betas', type=str)
    # configuration for LARS
    parser.add_argument('--lars_eps', type=float, default=1e-9)
    parser.add_argument('--lars_thresh', type=float, default=1e-2)

    # configuration of data loader
    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR10',
        choices=['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST', 'KMNIST'])
    parser.add_argument('--num_workers', type=int, default=7)
    # standard data augmentation
    parser.add_argument('--use_random_crop', action='store_true')
    parser.add_argument('--random_crop_padding', type=int, default=4)
    parser.add_argument('--use_horizontal_flip', action='store_true')
    # (dual-)cutout configuration
    parser.add_argument('--use_cutout', action='store_true', default=False)
    parser.add_argument(
        '--use_dual_cutout', action='store_true', default=False)
    parser.add_argument('--cutout_size', type=int, default=16)
    parser.add_argument('--cutout_prob', type=float, default=1)
    parser.add_argument('--cutout_inside', action='store_true', default=False)
    parser.add_argument('--dual_cutout_alpha', type=float, default=0.1)
    # random erasing configuration
    parser.add_argument(
        '--use_random_erasing', action='store_true', default=False)
    parser.add_argument('--random_erasing_prob', type=float, default=0.5)
    parser.add_argument(
        '--random_erasing_area_ratio_range', type=str, default='[0.02, 0.4]')
    parser.add_argument(
        '--random_erasing_min_aspect_ratio', type=float, default=0.3)
    parser.add_argument('--random_erasing_max_attempt', type=int, default=20)
    # mixup configuration
    parser.add_argument('--use_mixup', action='store_true', default=False)
    parser.add_argument('--mixup_alpha', type=float, default=1)
    # RICAP configuration
    parser.add_argument('--use_ricap', action='store_true', default=False)
    parser.add_argument('--ricap_beta', type=float, default=0.3)
    # label smoothing configuration
    parser.add_argument(
        '--use_label_smoothing', action='store_true', default=False)
    parser.add_argument('--label_smoothing_epsilon', type=float, default=0.1)
    # fp16
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    # use label processor
    parser.add_argument('--use_cl_lp', action='store_true')
    parser.add_argument('--lp_alpha',type=float, default=0.01)
    parser.add_argument('--lp_p', type=float, default=0.5)
    parser.add_argument('--is_select', action='store_true')
    parser.add_argument('--start_epoch',type = int ,default= 0)
    parser.add_argument('--end_epoch', type=int, default=140)
    # use regular label processor
    parser.add_argument('--use_rgl_cl_lp', action='store_true')
    parser.add_argument('--rgl_type',type=str,default='stable',help= "chose from stable、p_discrete1 , p_discrete2 and continuous")
    parser.add_argument('--rgl_interval', type=int, default=5)


    args = parser.parse_args()
    if not is_tensorboard_available:
        args.tensorboard = False
    if not is_apex_available:
        args.use_amp = False
    if args.use_amp:
        args.fp16 = True

    config = get_config(args)


    return config

