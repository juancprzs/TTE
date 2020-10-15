import sys
import torch
import random
import argparse
import numpy as np
import os.path as osp
import torch.backends.cudnn as cudnn

from utils.utils import (get_data_utils, AugWrapper, get_model, print_to_log, 
                         eval_chunk, eval_files)

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
    # Setup
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(DEVICE, args.seed)

    # Model
    model = get_model(args.experiment)

    # Parameters for Gaussian (if any)
    if (args.gauss_k is not None) and (args.gauss_s is not None):
        gauss_ps = (args.gauss_k, args.gauss_s)
    else:
        gauss_ps = None
    
    model_aug = AugWrapper(model, args.flip, args.n_crops, args.flip_crop, 
                           gauss_ps).to(DEVICE)

    # Print augmentations
    info = ','.join(model_aug.total_augs)
    print_to_log(info, args.info_log)

    # de-facto GPU usage will be increased by num of transforms!
    batch_size = int(args.batch_size / (1 + len(model_aug.total_augs)))

    # Data
    if args.num_chunk is None: # evaluate sequentially
        log_files = []
        for num_chunk in range(1, args.chunks+1):
            log_file = eval_chunk(model_aug, batch_size, args.chunks, num_chunk,
                                  DEVICE, args)
            log_files.append(log_file)

        eval_files(log_files, args.final_results)
    else: # evaluate a single chunk and exit
        log_file = eval_chunk(model_aug, adversary, batch_size, args.chunks, 
                              args.num_chunk, DEVICE, args)
        sys.exit()


if __name__ == "__main__":
    from utils.opts import parse_settings
    args = parse_settings()
    if args.files_eval is not None:
        eval_files(args.files_eval, args.final_results)
        sys.exit()
    else:
        main(args)
