import os
import torch
import random
import argparse
import numpy as np
import os.path as osp
from utils.resnet import ResNet18
import torch.backends.cudnn as cudnn

from utils.utils import (get_data_utils, AugWrapper, get_clean_acc, get_rob_acc, 
                         print_to_log, print_training_params)

# For deterministic behavior
cudnn.benchmark = False
cudnn.deterministic = True


def set_seed(device, seed=111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


def get_model(experiment):
    if experiment == 'local_trades':
        model = ResNet18(num_classes=10)
        state_dict = torch.load('./weights/local_trades_best.pth')['state_dict']
        state_dict = { k.replace('model.' ,'') : v for k, v in state_dict.items() }
        model.load_state_dict(state_dict, strict=False)
    elif experiment == 'trades':
        from experiments.trades import get_model
        model = get_model()

    return model

def main(args):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_name = osp.join(args.checkpoint, 'ckpt_eval.csv')
    set_seed(DEVICE, args.seed)

    # Model
    model = get_model(args.experiment)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Parameters for Gaussian (if any)
    if (args.gauss_k is not None) and (args.gauss_s is not None):
        gauss_ps = (args.gauss_k, args.gauss_s)
    else:
        gauss_ps = None
    
    model_aug = AugWrapper(model, args.flip, args.n_crops, args.flip_crop, 
                           gauss_ps).to(DEVICE)

    # de-facto GPU usage will be increased by num of transforms!
    batch_size = int(args.batch_size / (1 + len(model_aug.total_augs)))
    testloader = get_data_utils(args.test_samples, batch_size)
    # Print augmentations
    info = ','.join(model_aug.total_augs)
    print_to_log(info, log_name)

    clean_aug_acc = get_clean_acc(model_aug, testloader, DEVICE)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    rob_aug_acc = get_rob_acc(model_aug, testloader, DEVICE, batch_size, 
                              cheap=True, seed=args.seed)

    print_to_log(f'{clean_aug_acc:4.2f},{rob_aug_acc:4.2f}', log_name)


if __name__ == "__main__":
    from utils.opts import parse_settings
    args = parse_settings()

    # Log path: verify existence of checkpoint dir, or create it
    if not osp.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    # txt file with all params
    print_training_params(args, osp.join(args.checkpoint, 'params.txt'))

    main(args)
