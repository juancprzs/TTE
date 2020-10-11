import os
import torch
import random
import argparse
import numpy as np
import os.path as osp
from resnet import ResNet18
import torch.backends.cudnn as cudnn

from utils import (get_data_utils, AugWrapper, get_clean_acc, get_rob_acc, 
    print_to_log)

# For deterministic behavior
cudnn.benchmark = False
cudnn.deterministic = True


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


def main(args):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CKPT_NAME = 'local_trades_best.pth'
    set_seed(args.seed, DEVICE)

    testloader = get_data_utils()
    # Model
    resnet = ResNet18(num_classes=10).to(DEVICE)
    state_dict = torch.load(CKPT_NAME)['state_dict']
    state_dict = {k.replace('model.' ,'') : v for k, v in state_dict.items()}
    resnet.load_state_dict(state_dict, strict=False)

    std = torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1).to(DEVICE)
    mean = torch.tensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1).to(DEVICE)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    model_normal = AugWrapper(resnet, mean, std).to(DEVICE)
    print('Normal model:')
    clean_acc = get_clean_acc(model_normal, testloader, DEVICE)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    model_aug = AugWrapper(resnet, mean, std, augs=True, 
        n_crops=args.n_crops).to(DEVICE)
    print('Augmented model:')
    clean_aug_acc = get_clean_acc(model_aug, testloader, DEVICE)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    rob_aug_acc = get_rob_acc(model_aug, testloader, DEVICE, cheap=True)

    log_name = osp.join(args.checkpoint, 'ckpt_eval.csv')
    info = f'{clean_acc:4.2f},{clean_aug_acc:4.2f},{rob_aug_acc:4.2f}'
    print_to_log(info, log_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch TAR')
    parser.add_argument('--checkpoint', type=str, required=True,
        help='name of directory for saving results')
    parser.add_argument('--n-crops', type=int, default=0, 
        help='num of crops for augmentation')
    parser.add_argument('--seed', type=int, default=111, 
        help='for deterministic behavior')
    args = parser.parse_args()

    # Log path: verify existence of checkpoint dir, or create it
    if not osp.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    main(args)
