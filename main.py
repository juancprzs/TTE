import os
import torch
import random
import argparse
import numpy as np
import os.path as osp
import torch.backends.cudnn as cudnn

from utils.utils import (get_data_utils, AugWrapper, get_clean_acc, get_model,
                         compute_advs, print_to_log, print_training_params)

# For deterministic behavior
cudnn.benchmark = False
cudnn.deterministic = True


def set_seed(device, seed=111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


def main(args):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    testloader = get_data_utils(batch_size, args.test_samples, args.chunks, 
                                args.num_chunk)

    # Print augmentations
    info = ','.join(model_aug.total_augs)
    print_to_log(info, args.log_name)

    # Clean acc
    clean_acc = get_clean_acc(model_aug, testloader, DEVICE)

    # Rob acc
    advs, accs = compute_advs(model_aug, testloader, DEVICE, batch_size, 
                              cheap=args.cheap, seed=args.seed)

    # Send everything to file
    accs.update({ 'clean' : clean_acc })
    info = '\n'.join([f'{k}:{v:4.2f}' for k, v in accs.items()])
    print_to_log(info, args.log_name)
    print('Accuracies: \n', info)

    # Save adversaries
    torch.save(advs, osp.join(args.checkpoint, 'advs.pth')) # advs is a dict


if __name__ == "__main__":
    from utils.opts import parse_settings
    args = parse_settings()

    # Log path: verify existence of checkpoint dir, or create it
    if not osp.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    args.adv_dir = osp.join(args.checkpoint, 'advs')
    if not osp.exists(args.adv_dir):
        os.makedirs(args.adv_dir)

    # log
    args.log_name = osp.join(args.checkpoint, 'ckpt_eval.txt')

    # txt file with all params
    print_training_params(args, osp.join(args.checkpoint, 'params.txt'))

    main(args)
