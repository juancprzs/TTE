# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from adv_model import AdvImageNetModel
from resnet_model import (
    resnet_group, resnet_bottleneck, resnet_backbone)
from resnet_model import denoising

import tensorflow as tf
import numpy as np


NUM_BLOCKS = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}


class ResNetModel(AdvImageNetModel):
    def __init__(self, args):
        self.num_blocks = NUM_BLOCKS[args.depth]

    def get_logits(self, image):
        return resnet_backbone(image, self.num_blocks, resnet_group, resnet_bottleneck)


class ResNetDenoiseModel(AdvImageNetModel):
    def __init__(self, args):
        self.num_blocks = NUM_BLOCKS[args.depth]

    def get_logits(self, image):

        def group_func(name, *args):
            """
            Feature Denoising, Sec 6:
            we add 4 denoising blocks to a ResNet: each is added after the
            last residual block of res2, res3, res4, and res5, respectively.
            """
            l = resnet_group(name, *args)
            l = denoising(name + '_denoise', l, embed=True, softmax=True)
            return l

        return resnet_backbone(image, self.num_blocks, group_func, resnet_bottleneck)


class ResNeXtDenoiseAllModel(AdvImageNetModel):
    """
    ResNeXt 32x8d that performs denoising after every residual block.
    """
    def __init__(self, args):
        self.num_blocks = NUM_BLOCKS[args.depth]

    def get_logits(self, image):

        def block_func(l, ch_out, stride):
            """
            Feature Denoising, Sec 6.2:
            The winning entry, shown in the blue bar, was based on our method by using
            a ResNeXt101-32×8 backbone
            with non-local denoising blocks added to all residual blocks.
            """
            l = resnet_bottleneck(l, ch_out, stride, group=32, res2_bottleneck=8)
            l = denoising('non_local', l, embed=False, softmax=False)
            return l

        return resnet_backbone(image, self.num_blocks, resnet_group, block_func)


# =================================== #
#            MODEL WRAPPER            #
# =================================== #


class ModelWrapper(ResNetDenoiseModel):
    def __init__(self, args, model):
        self.model = model
        self.transforms = AUMENTS(args.flip, args.n_crops, args.flip_crop)

    def get_logits(self, image):
        print(image.get_shape())  # it is 224 x 224 at the moment?
        mean_logits = tf.zeros((tf.shape(image)[0], 1000))
        n = len(self.transforms)
        for t in self.transforms:
            mean_logits = tf.add(self.model.get_logits(t.forward(image)), mean_logits)

        # # condition function
        # n = len(self.transforms)
        # cond = lambda i, x: i < n

        # # body function
        # body = lambda i, x: (i + 1,
        #                      self.model.get_logits(self.transforms[i].forward(image)) + x)

        # # for loop
        # _, mean_logits = tf.while_loop(cond, body, (0, mean_logits))

        mean_logits = mean_logits / tf.constant(n, dtype=tf.float32)

        return mean_logits


def crop(image, orig_x, orig_y, crop_size):
    # image = tf.slice(image, [0, 0, orig_x, orig_y], [image.get_shape()[0], image.get_shape()[1], crop_size, crop_size])
    # image = tf.transpose(image, [0, 2, 3, 1])
    # image = tf.image.crop_to_bounding_box(image, orig_x, orig_y, crop_size, crop_size)
    # image = tf.transpose(image, [0, 3, 1, 2])
    print(image.get_shape())  # it is 224 x 224 at the moment?
    image = image[:, :, orig_x: (orig_x + crop_size), orig_y: (orig_y + crop_size)]
    print(image.get_shape())
    return image


class Diffidentity():

    def forward(self, image):
        '''
        Performs the Center crop of 224.
        Expected input of 256 x 256
        '''
        with tf.variable_scope('Identity'):
            # indent_image = image[:, :, (128 - 112): (128 + 112), (128 - 112): (128 + 112)]
            # indent_image = image
            indent_image = crop(image, 14, 14, 224)
        return indent_image


class DiffFlip():
    def forward(self, image):
        '''
        Performs the Center crop of 224.
        Expected input of 256 x 256
        '''
        # images = image[:, :, 128 - 112: 128 + 112, 128 - 112: 128 + 112]
        image = crop(image, 14, 14, 224)
        return tf.reverse(image, axis=tf.constant([3,], dtype=tf.int32), name='Flip')


class DiffCrop():
    def __init__(self, crop_size=224, pad_size=0, input_size=256):
    # def __init__(self, crop_size=224, pad_size=28, input_size=224):
        '''
        I have chosen 28 for the pad 
        '''
        self.pad = [[0, 0],
                    [0, 0],
                    [pad_size, pad_size],
                    [pad_size, pad_size]]
        valid_init_limit = input_size + int(2*pad_size) - crop_size
        self.orig_x = np.random.randint(valid_init_limit)
        self.orig_y = np.random.randint(valid_init_limit)
        # get ends for x and y
        self.end_x = self.orig_x + crop_size
        self.end_y = self.orig_y + crop_size
        self.crop_size = crop_size

    def forward(self, image):
        with tf.variable_scope('Crop'):
            # paddings = tf.constant(self.pad)
            # crop_image = tf.pad(image, paddings, 'REFLECT')
            # image = tf.pad(image, paddings, 'CONSTANT')
            # crop_image = crop_image[:, :, self.orig_x:self.end_x, self.orig_y:self.end_y]
            crop_image = crop(image, self.orig_x, self.orig_y, self.crop_size)
        return crop_image


class DiffFlipCrop():
    def __init__(self, superclass):
        self.pad = superclass.pad
        self.orig_y = superclass.orig_y
        self.orig_x = superclass.orig_x
        self.end_x = superclass.end_x
        self.end_y = superclass.end_y
        self.crop_size = superclass.crop_size

    def forward(self, image):

        with tf.variable_scope('FlipCrop'):
            flip_image = tf.reverse(image, axis=tf.constant([3,], dtype=tf.int32))
            # paddings = tf.constant(self.pad)
            # fc_image = tf.pad(flip_image, paddings, 'CONSTANT')
            # fc_image = fc_image[:, :, self.orig_x:self.end_x, self.orig_y:self.end_y]
            crop_image = crop(flip_image, self.orig_x, self.orig_y, self.crop_size)
        return fc_image


def AUMENTS(flip, n_crops, flip_crop):
    augments = [Diffidentity()]

    if flip:
        augments.append(DiffFlip())

    if n_crops != 0:
        crops_fs = [DiffCrop() for _ in range(n_crops)]
        augments.extend(crops_fs)

    if flip and (n_crops != 0) and flip_crop:
        fc_augms = [DiffFlipCrop(crop) for crop in crops_fs]
        augments.extend(fc_augms)

    print(f'Using {len(augments)} augmentations')

    return augments
