import os
import cv2
import tqdm
import torch
import argparse
import numpy as np

from tensorpack import TowerContext
from tensorpack.tfutils import get_model_loader

import tensorflow as tf

from torchvision import transforms, datasets

from autoattack import AutoAttack, utils_tf

import nets


def arguments():
    parser = argparse.ArgumentParser()
    # path args
    parser.add_argument('--load', type=str, default='weights/R152-Denoise.npz',
                        help='Path to a model to load for evaluation.')
    parser.add_argument('--data', type=str, default='/home/gjeanneret/imagenet/dataset',
                        help='Path to imagenet folder')
    parser.add_argument('--output-path', default='runs', type=str,
                        help='Output path.')

    # chunks evaluation
    parser.add_argument('--total-chunks', default=1, type=int,
                        help='Total number of chunks, must not change along all scripts')
    parser.add_argument('--actual-chunk', default=0, type=int,
                        help='Actual chunk to evaluate. Values: 0 <= actual < total-chunks')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size')

    # Model args
    parser.add_argument('-d', '--depth', type=int, default=152, choices=[50, 101, 152],
                        help='ResNet depth')
    parser.add_argument('--arch', help='Name of architectures defined in nets.py',
                        default='ResNetDenoise')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--n-crops', default=0, type=int)
    parser.add_argument('--flip-crop', action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    # Attack options
    parser.add_argument('--epsilon', type=float, default=8 / 255,
                        help='Epsilon for the attack')

    # Miscellaneous
    parser.add_argument('--save-adv', action='store_true',
                        help='Save adversarial images')
    parser.add_argument('--evaluate-chunks', action='store_true',
                        help='Evaluate the chunks on --output-path')

    return parser.parse_args()


# ========================================
# Chunks dataset


class ChunkDataset():
    def __init__(self, dataset, num_chunk=0, total_chunks=1):
        self.dataset = dataset
        self.indexes = [i for i in range(len(dataset)) if (i % total_chunks) == num_chunk]
        self.len = len(self.indexes)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data_idx = self.indexes[idx]
        return self.dataset[data_idx]


class TORCH_TO_TF():
    def __call__(self, img):
        return torch.flip(img, (0,))


def compute_corrects(features, label):
    '''
    features numpy array: B x 1000
    label numpy vector: B
    '''

    pred = np.argmax(features, axis=1)
    correct = pred == label
    return np.sum(correct), pred


def loader(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    return im


def main(args):


    np.random.seed(args.seed)

    os.makedirs(args.output_path, exist_ok=True)

    # Create Model
    model = getattr(nets, args.arch + 'Model')(args)
    model = nets.ModelWrapper(args, model)

    # forward here
    img_input = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3])
    image_ = tf.transpose(img_input, [0, 3, 1, 2])
    lab_input = tf.placeholder(dtype=tf.int64, shape=[None])
    with TowerContext('', is_training=False):
        logits = model.get_logits(image_)

    # Session and load weights
    sess = tf.Session()
    get_model_loader(args.load).init(sess)

    model_adapted = utils_tf.ModelAdapter(logits, img_input, lab_input, sess, num_classes=1000)

    # load dataset
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])  # the model normalizes everything inside
    val_transform = transforms.Compose([
            transforms.Resize(256),  # Shortest edge will become 256
            transforms.CenterCrop(256),  # CenterCrop of 256, the model internally takes the 224 x 224 crop
            transforms.ToTensor(),
            normalize,
            TORCH_TO_TF()  # The model takes as input a BGR image
        ])
    dataset = datasets.ImageFolder(os.path.join(args.data, 'val'), transform=val_transform)
    chunkloader = ChunkDataset(dataset, num_chunk=args.actual_chunk,
                               total_chunks=args.total_chunks)
    dataloader = torch.utils.data.DataLoader(chunkloader, batch_size=args.batch,
                                             shuffle=False, num_workers=8,
                                             pin_memory=True)

    print('\n\n\nTOTAL IMAGES ON THIS CHUNK: {}'.format(len(chunkloader)))

    # Create adversary
    adversary = AutoAttack(model_adapted, norm='Linf', eps=args.epsilon,
                           version='standard', is_tf_model=True, verbose=False)

    # Count variables
    n = 0
    c_clean = 0
    c_adv = 0

    if args.save_adv:
        labels = []
        pred = []
        advers_pred = []
        adv_examples = []

    for clean_x, label in tqdm.tqdm(dataloader):

        numpy_clean = np.transpose(clean_x.numpy(), [0, 2, 3, 1])

        # compute clean acc
        features = sess.run(logits, feed_dict={img_input: numpy_clean})
        clean_correct, clean_pred = compute_corrects(features, label.numpy())

        # compute adversary images
        adv_x = adversary.run_standard_evaluation(clean_x, label, bs=args.batch)

        numpy_adv = np.transpose(adv_x.detach().cpu().numpy(), [0, 2, 3, 1])

        # compute adversary acc
        adv_features = sess.run(logits, feed_dict={img_input: numpy_adv})
        adv_correct, adv_pred = compute_corrects(adv_features, label.numpy())

        c_clean += clean_correct
        c_adv += adv_correct
        n += label.size(0)

        if args.save_adv:
            labels.append(label)
            pred.append(torch.tensor(clean_pred))
            advers_pred.append(torch.tensor(adv_pred))
            adv_examples.append(torch.tensor(numpy_adv))

    with open(f'{args.output_path}/results-chunk{args.actual_chunk}-total-chunks-{args.total_chunks}.txt', 'w') as f:
        f.write(f'n:{n}\nclean corrects:{c_clean}\nadversary correct:{c_adv}')
    print(f'n:{n}\nclean corrects:{c_clean}\nadversary correct:{c_adv}')

    if args.save_adv:
        torch.save({
            'gt': torch.cat(labels, dim=0),
            'clean pred': torch.cat(pred, dim=0),
            'adv pred': torch.cat(advers_pred, dim=0),
            'adv examples': torch.cat(adv_examples, dim=0),
            }, f'{args.output_path}/data-chunk{args.actual_chunk}-total-chunks-{args.total_chunks}.pth')


def evaluate_chunks(args):
    existing_files = []
    missing_files = []

    for i in range(args.total_chunks):
        path = os.path.join(args.output_path,
                            f'results-chunk{i}-total-chunks-{args.total_chunks}.txt')

        if os.path.exists(path):
            existing_files.append(path)
        else:
            missing_files.append(i)

    if len(missing_files) != 0:
        print('Missing experiments:', missing_files)

    clean = 0
    adv =  0
    n = 0

    for file in existing_files:
        with open(file, 'r') as f:
            data = f.read()

        for line in data.split('\n'):
            k, v = line.split(':')
            v = int(v)
            if k == 'n':
                n += v
            elif k == 'clean corrects':
                clean += v
            elif k == 'clean adversary':
                adv += v

    message = f'Results:\n\tTop1: {100 * clean / n}\n\tTop1 adv: {100 * adv / n}'

    print(message)
    with open(f'{args.output_path}/final-results.txt', 'w') as f:
        f.write(message)

if __name__ == '__main__':
    
    args = arguments()

    if args.evaluate_chunks:
        evaluate_chunks(args)
    else:
        main(args)
