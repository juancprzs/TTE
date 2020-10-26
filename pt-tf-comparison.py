#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import numpy as np
import cv2
import tensorflow as tf

from tensorpack import TowerContext
from tensorpack.tfutils import get_model_loader
from tensorpack.dataflow.dataset import ILSVRCMeta

# tf models
import mid_nets as nets

import torch


# conv model
def conv(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.conv1(x)


"""
A small inference example for attackers to play with.
"""


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--depth', help='ResNet depth',
                    type=int, default=152, choices=[50, 101, 152])
parser.add_argument('--arch', help='Name of architectures defined in nets.py',
                    default='ResNetDenoise')
parser.add_argument('-b', '--block', default=3, type=int, choices=[0, 1, 2, 3, 4, 5],
                    help='Get feautres up to that lock. 5 get the prediction')
parser.add_argument('--load', help='path to checkpoint')
parser.add_argument('--input', help='path to input image')
args = parser.parse_args()


sample = np.ones((3, 10, 10)).astype('float32')

# ======================
# tf

model = getattr(nets, args.arch + 'Model')(args)

input = tf.placeholder(tf.float32, shape=(None, 3, 10, 10))

with TowerContext('', is_training=False):
    logits = model.get_logits(image)

sess = tf.Session()
get_model_loader(args.load).init(sess)

tf_f = sess.run(logits, feed_dict={input: np.array([sample])})


# =======================
# pt

W = dict(np.load(args.load))['conv0/W']

torch_model = conv()
# load weight differntly
torch_model.conv1.weight = torch.tensor(np.transpose(np_weights, (3, 2, 0, 1)), dtype=torch.float)

sample_pt = torch.ones(1, 3, 10, 10)
pt_f = torch_model(sample_pt)

import ipdb; ipdb.set_trace()