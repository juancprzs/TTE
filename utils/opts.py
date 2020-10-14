from argparse import ArgumentParser

def parse_settings():
    EXP_CHOICES = ['local_trades','trades']
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
    parser.add_argument('--test-samples', type=int, default=None, 
                        help='num of test instances to use')
    args = parser.parse_args()

    return args
