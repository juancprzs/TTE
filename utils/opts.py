import os
import os.path as osp
from argparse import ArgumentParser
from utils.utils import print_training_params

def parse_settings():
    EXP_CHOICES = ['local_trades','trades','awp']
    parser = ArgumentParser(description='PyTorch code for TAR: Test-time '
                            'Augmentation for Robustness')
    parser.add_argument('--experiment', type=str, default='trades', 
                        help='method on which to test TAR', choices=EXP_CHOICES)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='name of directory for saving results')
    parser.add_argument('--n-crops', type=int, default=0, 
                        help='num of crops for aug')
    parser.add_argument('--flip', action='store_true', default=False,
                        help='whether to use flip aug')
    parser.add_argument('--flip-crop', action='store_true', default=False,
                        help='whether to combine flip aug with crops')
    parser.add_argument('--gauss-k', type=int, default=None, 
                        help='kernel size for Gaussian aug')
    parser.add_argument('--gauss-s', type=float, default=None, 
                        help='variance for Gaussian aug')
    parser.add_argument('--seed', type=int, default=0, 
                        help='for deterministic behavior')
    parser.add_argument('--batch-size', type=int, default=250,
                        help='batch size')
    parser.add_argument('--cheap', action='store_true', default=False,
                        help='whether to use cheap attack (useful for debug)')
    parser.add_argument('--chunks', type=int, default=10, 
                        help='num of chunks in which to break the dataset')
    parser.add_argument('--num-chunk', type=int, default=None, 
                        help='index of chunk to evaluate on')
    parser.add_argument('--eval-files', action='store_true', default=False,
		                help='evaluate based on files at '
                             'checkpoint/logs/results_chunk*of*_*to*.txt')
    args = parser.parse_args()

    # Log path: verify existence of checkpoint dir, or create it
    if not osp.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    args.adv_dir = osp.join(args.checkpoint, 'advs')
    if not osp.exists(args.adv_dir):
        os.makedirs(args.adv_dir)

    args.logs_dir = osp.join(args.checkpoint, 'logs')
    if not osp.exists(args.logs_dir):
        os.makedirs(args.logs_dir)

    # txt file with all params
    chunk = 'all' if args.num_chunk is None else args.num_chunk
    args.info_log = osp.join(args.checkpoint, f'info_chunk_{chunk}.txt')
    print_training_params(args, args.info_log)

    # final results
    args.final_results = osp.join(args.checkpoint, f'results.txt')

    return args
