import os
import random
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
from scipy import ndimage
from autoattack import AutoAttack

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import TensorDataset
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader, random_split


class DiffCrop(nn.Module):
    def __init__(self, inp_size=32, crop_size=32, pad_size=4):
        super(DiffCrop, self).__init__()
        self.pad = tuple([pad_size for _ in range(4)]) # udlr
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


class GaussianLayer(nn.Module):
    # Code taken from (and slightly modified)
    # https://discuss.pytorch.org/t/gaussian-kernel-layer/37619
    def __init__(self, kernel_size=5, sigma=1):
        super(GaussianLayer, self).__init__()
        assert kernel_size % 2 != 0, 'kernel_size should be odd'
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.seq = nn.Sequential(
                                 nn.ReflectionPad2d(self.kernel_size // 2), 
                                 nn.Conv2d(3, 3, kernel_size=self.kernel_size, 
                                           stride=1, padding=0, bias=None, 
                                           groups=3)
        )
        self.weights_init()
    
    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n = np.zeros((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        n[center, center] = 1
        k = ndimage.gaussian_filter(n, sigma=self.sigma)
        for _, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))

class NormalizedWrapper(nn.Module):
    def __init__(self, model, mean=None, std=None):
        super(NormalizedWrapper, self).__init__()
        self.model = model
        self.normalize = False
        if mean is None:
            assert std is None
            self.normalize = True
            # std
            std = torch.tensor(std).view(1, 3, 1, 1)
            self.std = nn.Parameter(std, requires_grad=False)
            # mean
            mean = torch.tensor(mean).view(1, 3, 1, 1)
            self.mean = nn.Parameter(mean, requires_grad=False)

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        return self.model(x)


class AugWrapper(nn.Module):
    def __init__(self, model, flip=False, n_crops=0, flip_crop=False, 
            gauss_ps=None):
        super(AugWrapper, self).__init__()
        self.model = model
        # transforms
        self.transforms = [lambda x: x] # the identity
        self.total_augs = self._init_augs(flip, n_crops, gauss_ps, flip_crop)
        
        if len(self.total_augs) != 0: # whether augmentations are used
            print('Using augmentations: ' + ','.join(self.total_augs))
        else:
            print('NOT using augmentations!')

        print(f'{len(self.transforms)} transforms: {self.transforms}')
    
    def _init_augs(self, flip, n_crops, gauss_ps, flip_crop):
        total_augs = []
        if flip:
            total_augs.append('flip')
            # flip augmentations
            flip_f = DiffFlip()
            self.transforms.append(lambda x: flip_f(x))
        
        if n_crops != 0:
            total_augs.append(f'crops n={n_crops}')
            # crop augmentations
            crops_fs = [DiffCrop() for _ in range(n_crops)]
            self.transforms.extend([lambda x: f(x) for f in crops_fs])
        
        if flip and (n_crops != 0) and flip_crop:
            total_augs.append(f'flipped-crops n={n_crops}')
            # flip-crop augmentations
            self.transforms.extend([lambda x: f(flip_f(x)) for f in crops_fs])
        
        if gauss_ps is not None:
            kernel_size, sigma = gauss_ps
            total_augs.append(f'gauss k={kernel_size}, s={sigma}')
            self.gauss_layer = GaussianLayer(kernel_size=kernel_size, 
                                             sigma=sigma)
            self.transforms.append(lambda x: self.gauss_layer(x))

        return total_augs


    def forward(self, x):
        x = torch.cat([t(x).unsqueeze(0) for t in self.transforms])
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        scores = self.model(x)
        scores = scores[0] if isinstance(scores, tuple) else scores # resnet case
        scores = scores.view(len(self.transforms), -1, scores.size(1))
        scores = torch.mean(scores, dim=0) # average across augmentations
        return scores


def get_data_utils(test_samples=None, batch_size=50):
    testset = CIFAR10(root='./data', train=False, download=True,
                      transform=Compose([ToTensor()]))
    tot_instances = len(testset)
    if (test_samples is not None) and (test_samples < tot_instances):
        remaining = tot_instances - test_samples
        assert remaining > 0
        print(f'Using {test_samples} samples only!')
        generator = torch.Generator().manual_seed(111) # for reproducibility
        testset, _ = random_split(testset, [test_samples, remaining], 
                                  generator=generator)
    
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


def get_rob_acc(model, testloader, device, batch_size, cheap=False, seed=0):
    model.eval()
    adversary = AutoAttack(model.forward, norm='Linf', eps=0.031, verbose=True)
    adversary.seed = seed
    if cheap:
        print('Running CHEAP attack')
        # based on
        # https://github.com/fra31/auto-attack/blob/master/autoattack/autoattack.py#L230
        # adversary.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
        adversary.attacks_to_run = ['apgd-ce', 'square']
        adversary.apgd.n_iter = 2
        adversary.apgd.n_restarts = 1
        adversary.fab.n_restarts = 1
        adversary.apgd_targeted.n_restarts = 1
        adversary.fab.n_target_classes = 2
        adversary.apgd_targeted.n_target_classes = 2
        adversary.square.n_queries = 2

    imgs = torch.cat([x for (x, y) in testloader], 0)[:600]
    labs = torch.cat([y for (x, y) in testloader], 0)[:600]
    advs = adversary.run_standard_evaluation_individual(imgs, labs, 
                                                        bs=batch_size)


    adversary = AutoAttack(model.forward, norm='Linf', eps=0.031, verbose=True)
    if cheap:
        print('Running CHEAP attack')
        # based on
        # https://github.com/fra31/auto-attack/blob/master/autoattack/autoattack.py#L230
        # adversary.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
        adversary.attacks_to_run = ['apgd-ce', 'square']
        adversary.apgd.n_iter = 2
        adversary.apgd.n_restarts = 1
        adversary.fab.n_restarts = 1
        adversary.apgd_targeted.n_restarts = 1
        adversary.fab.n_target_classes = 2
        adversary.apgd_targeted.n_target_classes = 2
        adversary.square.n_queries = 2
    other_advs = adversary.run_standard_evaluation(imgs, labs, bs=batch_size)
    
    accs = compute_accs(model, advs, labs, device, batch_size)

    import pdb; pdb.set_trace()
    return advs, accs

def compute_accs(model, advs, labels, device, batch_size):
    accs = {}
    all_preds = []
    for attack_name, curr_advs in advs.items():
        dataset = TensorDataset(curr_advs, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=1, pin_memory=True, drop_last=False)
        total_corr = 0
        curr_preds = []
        with torch.no_grad():
            for img, lab in dataloader:
                img, lab = img.to(device), lab.to(device)
                output = model(img)
                pred = output.max(1)[1]
                curr_preds.append(pred)
                import pdb; pdb.set_trace()
                total_corr += (pred == lab).sum().item()

        curr_preds = torch.cat(curr_preds)
        all_preds.append(curr_preds)
            
        curr_acc = 100. * total_corr / labels.size(0)
        accs.update({ attack_name : curr_acc })

    return accs


def print_to_log(text, txt_file_path):
    with open(txt_file_path, 'a') as text_file:
        print(text, file=text_file)


def print_training_params(args, txt_file_path):
    d = vars(args)
    text = ' | '.join([str(key) + ': ' + str(d[key]) for key in d])
    # Print to log and console
    print_to_log(text, txt_file_path)
    print(text)
