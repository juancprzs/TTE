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

import mid_nets as nets

"""
A small inference example for attackers to play with.
"""


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--depth', help='ResNet depth',
                    type=int, default=152, choices=[50, 101, 152])
parser.add_argument('--arch', help='Name of architectures defined in nets.py',
                    default='ResNetDenoise')
parser.add_argument('-b', '--block', default=5, type=int, choices=[0, 1, 2, 3, 4, 5],
                    help='Get feautres up to that lock. 5 get the prediction')
parser.add_argument('--load', help='path to checkpoint')
parser.add_argument('--input', help='path to input image')
args = parser.parse_args()

model = getattr(nets, args.arch + 'Model')(args)

input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
image = tf.transpose(input, [0, 3, 1, 2])
image =  image / 255
with TowerContext('', is_training=False):
    logits = model.get_logits(image)

sess = tf.Session()
get_model_loader(args.load).init(sess)

sample = cv2.imread(args.input)  # this is a BGR image, not RGB
# imagenet evaluation uses standard imagenet pre-processing
# (resize shortest edge to 256 + center crop 224).
# However, for images of unknown sources, let's just do a naive resize.
# sample = cv2.resize(sample, (224, 224))
import pdb; pdb.set_trace()

features = sess.run(logits, feed_dict={input: np.array([sample])})
np.savez('features_block{}.npz'.format(args.block), *features)