#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import cv2
import glob
import numpy as np
import os
import socket
import sys

import horovod.tensorflow as hvd

from tensorpack import *
from tensorpack.tfutils import SmartInit

import nets
from adv_model import NoOpAttacker, PGDAttacker
from third_party.imagenet_utils import get_val_dataflow, eval_on_ILSVRC12
from third_party.utils import HorovodClassificationError


def create_eval_callback(name, tower_func, condition):
    """
    Create a distributed evaluation callback.

    Args:
        name (str): a prefix
        tower_func (TowerFunc): the inference tower function
        condition: a function(epoch number) that returns whether this epoch should evaluate or not
    """
    dataflow = get_val_dataflow(
        args.data, args.batch,
        num_splits=hvd.size(), split_index=hvd.rank())
    # We eval both the classification error rate (for comparison with defenders)
    # and the attack success rate (for comparison with attackers).
    infs = [HorovodClassificationError('wrong-top1', '{}-top1-error'.format(name)),
            HorovodClassificationError('wrong-top5', '{}-top5-error'.format(name)),
            HorovodClassificationError('attack_success', '{}-attack-success-rate'.format(name))
            ]
    cb = InferenceRunner(
        QueueInput(dataflow), infs,
        tower_name=name,
        tower_func=tower_func).set_chief_only(False)
    cb = EnableCallbackIf(
        cb, lambda self: condition(self.epoch_num))
    return cb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='Path to a model to load for evaluation or resuming training.')
    parser.add_argument('--logdir', help='Directory suffix for models and training stats.')

    # run on a directory of images:
    parser.add_argument('--data', help='ILSVRC dataset dir', default='/home/gjeanneret/imagenet/dataset')
    parser.add_argument('--fake', help='Use fakedata to test or benchmark this model', action='store_true')
    parser.add_argument('--no-zmq-ops', help='Use pure python to send/receive data',
                        action='store_true')
    parser.add_argument('--batch', help='Per-GPU batch size', default=32, type=int)

    # attacker flags:
    parser.add_argument('--attack-iter', help='Adversarial attack iteration',
                        type=int, default=30)
    parser.add_argument('--attack-epsilon', help='Adversarial attack maximal perturbation',
                        type=float, default=16.0)
    parser.add_argument('--attack-step-size', help='Adversarial attack step size',
                        type=float, default=1.0)
    parser.add_argument('--use-fp16xla',
                        help='Optimize PGD with fp16+XLA in training or evaluation. '
                        '(Evaluation during training will still use FP32, for fair comparison)',
                        action='store_true')

    # architecture flags:
    parser.add_argument('-d', '--depth', help='ResNet depth',
                        type=int, default=152, choices=[50, 101, 152])
    parser.add_argument('--arch', help='Name of architectures defined in nets.py',
                        default='ResNetDenoise')

    # augment
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--n-crops', default=0, type=int)
    parser.add_argument('--flip-crop', action='store_true')
    parser.add_argument('--output-path', default='runs', type=str)
    parser.add_argument('--exp-name', type=str)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    args.eval = True

    np.random.seed(args.seed)

    # Define model
    model = getattr(nets, args.arch + 'Model')(args)
    model = nets.ModelWrapper(args, model)

    # Define attacker
    if args.attack_iter == 0:
        attacker = NoOpAttacker()
    else:
        attacker = PGDAttacker(
            args.attack_iter, args.attack_epsilon, args.attack_step_size,
            prob_start_from_clean=0.2 if not args.eval else 0.0)
        if args.use_fp16xla:
            attacker.USE_FP16 = True
            attacker.USE_XLA = True
    model.set_attacker(attacker)

    hvd.init()

    sessinit = SmartInit(args.load)

    os.makedirs(args.output_path, exist_ok=True)

    # single-GPU eval, slow
    # ds = get_val_dataflow(args.data, args.batch)
    ds = get_val_dataflow(args.data, args.batch, augmentors='wrapper')
    eval_on_ILSVRC12(model, sessinit, ds, args.output_path + '/' + args.exp_name)

# CUDA_VISIBLE_DEVICES=0 python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --exp-name baseline-2.txt