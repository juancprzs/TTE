import os
import random
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
from autoattack import AutoAttack

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor


class DiffCrop(nn.Module):
    def __init__(self, inp_size=32, crop_size=32, pad_size=4):
        super(DiffCrop, self).__init__()
        self.pad = tuple([pad_size for _ in range(4)])
        # get origins for x and y
        valid_init_limit = inp_size + int(2*pad_size) - crop_size
        self.orig_x = np.random.randint(valid_init_limit)
        self.orig_y = np.random.randint(valid_init_limit)
        # get ends for x and y
        self.end_x = self.orig_x + crop_size
        self.end_y = self.orig_y + crop_size
    
    def forward(self, x):
        x = F.pad(x, pad=self.pad) # pad input
        x = x[:, :, self.orig_x:self.end_x, self.orig_y:self.end_y] # crop it
        return x


class DiffFlip(nn.Module):
    def __init__(self):
        super(DiffFlip, self).__init__()

    def forward(self, x):
        return x.flip(3) # 3 = the left-right dim


class AugWrapper(nn.Module):
    def __init__(self, model, mean, std, n_crops=2, augs=False):
        super(AugWrapper, self).__init__()
        self.augs = augs
        self.model = model
        self.std = nn.Parameter(std, requires_grad=False)
        self.mean = nn.Parameter(mean, requires_grad=False)
        # all transforms
        self.transforms = [lambda x: x] # the identity
        if self.augs:
            # crop augmentations
            crops_fs = [DiffCrop() for _ in range(n_crops)]
            self.transforms.extend([lambda x: f(x) for f in crops_fs])
            # flip augmentations
            flip_f = DiffFlip()
            self.transforms.append(lambda x: flip_f(x))
            # flip-crop augmentations
            self.transforms.extend([lambda x: f(flip_f(x)) for f in crops_fs])

        print(f'{len(self.transforms)} transforms: {self.transforms}')
    
    def forward(self, x):
        x = (x - self.mean) / self.std
        x = torch.cat([t(x).unsqueeze(0) for t in self.transforms])
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        scores = self.model(x)[0]
        scores = scores.view(len(self.transforms), -1, scores.size(1))
        scores = torch.mean(scores, dim=0) # average across augmentations
        return scores


def get_data_utils(batch_size=50):
    testset = CIFAR10(root='./data', train=False, 
        transform=Compose([ToTensor()]), download=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=True, drop_last=False)

    return testloader


def get_clean_acc(model, testloader, device):
    model.eval()
    n, total_acc = 0, 0
    with torch.no_grad():
        for X, y in tqdm(testloader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            total_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    
    acc = 100. * total_acc / n
    print(f'Clean accuracy: {acc:.4f}')
    return acc


def get_rob_acc(model, testloader, device, cheap=False):
    model.eval()
    adversary = AutoAttack(model.forward, norm='Linf', eps=0.031, verbose=False)
    adversary.seed = 0
    if cheap:
        print('Running CHEAP attack')
        adversary.attacks_to_run = ['apgd-ce'] # , 'apgd-dlr', 'square', 'fab']
        adversary.apgd.n_iter = 5

    n, total_acc = 0, 0
    pbar = tqdm(testloader)
    for _, (X, y) in enumerate(pbar):
        X, y = X.to(device), y.to(device)
        x_adv = adversary.run_standard_evaluation(X, y, bs=y.size(0))
        with torch.no_grad():
            output = model(x_adv)
            total_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
        acc = 100. * total_acc / n
        pbar.set_description(f'Current robust acc: {acc:.4f}')

    acc = 100. * total_acc / n
    print(f'Robust acc: {acc:.4f}')
    return acc


def print_to_log(text, txt_file_path):
    with open(txt_file_path, 'a') as text_file:
        print(text, file=text_file)


def print_training_params(args, txt_file_path):
    d = vars(args)
    text = ' | '.join([str(key) + ': ' + str(d[key]) for key in d])
    # Print to log and console
    print_to_log(text, txt_file_path)
    print(text)
