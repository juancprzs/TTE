import os
import cv2
import tqdm
import torch
import argparse
import numpy as np
import PIL.Image as Image

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
    parser.add_argument('--seed', type=int, default=1)

    # Attack options
    parser.add_argument('--epsilon', type=float, default=16 / 255,
                        help='Epsilon for the attack')

    # Miscellaneous
    parser.add_argument('--save-adv', action='store_true',
                        help='Save adversarial images')
    parser.add_argument('--evaluate-chunks', action='store_true',
                        help='Evaluate the chunks on --output-path')

    return parser.parse_args()


# ========================================
# Chunks dataset


class DataLoader():
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n = 0

        self.n_iters = len(dataset) / batch_size
        self.n_iters = int(self.n_iters) + 1 if (self.n_iters % 1) != 0 else int(self.n_iters)

    def __len__(self):
        return self.n_iters

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):

        if self.n >= self.n_iters:
            raise StopIteration

        current = self.n * self.batch_size
        next = current + self.batch_size

        data = [self.dataset[idx] for idx in range(current, min(next, len(self.dataset)))]
        images = np.concatenate([data[idx][0] for idx in range(len(data))], axis=0)
        labels = np.concatenate([data[idx][1] for idx in range(len(data))], axis=0)

        self.n += 1

        return images, labels


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


def compute_corrects_advs(advs_imgs, label, sess, placeholder, logits):
    current_corrects = {}
    all_preds = []

    for attack_name, advs in advs_imgs.items():

        numpy_adv = np.transpose(advs.detach().cpu().numpy(), [0, 2, 3, 1])
        adv_features = sess.run(logits, feed_dict={placeholder: numpy_adv})
        corrects, pred = compute_corrects(adv_features, label)

        current_corrects[attack_name] = {}
        current_corrects[attack_name]['corrects'] = corrects
        current_corrects[attack_name]['pred'] = pred
        all_preds.append(pred)

    # compute robust acc
    all_preds = np.concatenate([np.expand_dims(x, axis=0) for x in all_preds])
    temp_labels = np.expand_dims(label, axis=0)
    temp_labels = np.concatenate([temp_labels for _ in range(len(advs_imgs))])
    where_all_correct = np.prod(np.equal(all_preds, temp_labels).astype('long'), axis=0)
    current_corrects['rob acc'] = np.sum(where_all_correct)
    return current_corrects


def load_and_process(path, inter=cv2.INTER_CUBIC):
    # import pdb; pdb.set_trace()
    shortest = 256

    image = cv2.imread(path)
    
    # resize the image
    dim = None
    (h, w) = image.shape[:2]

    H = False

    if h > w:
        r = shortest / w
        dim = (shortest, int(h * r))
    elif w > h:
        H = True
        r = shortest / h
        dim = (int(w * r), shortest)
        # dim = (shortest, int(w * r))
    else:
        resized =  cv2.resize(image, (shortest, shortest), interpolation=inter)
        return np.expand_dims(resized, axis=0)

    resized = cv2.resize(image, dim, interpolation=inter)

    # center crop
    (h, w) = resized.shape[:2]

    if H:
        center_crop = resized[:, (int(w / 2) - 128): (int(w / 2) + 128), :]
    else:
        center_crop = resized[(int(h / 2) - 128): (int(h / 2) + 128), :, :]

    center_crop = np.expand_dims(center_crop, axis=0)
    
    return center_crop


class ImageNetDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        target = np.array([target])
        image = load_and_process(path)
        return image, target


def main(args):


    np.random.seed(args.seed)

    os.makedirs(args.output_path, exist_ok=True)

    # Create Model
    model = getattr(nets, args.arch + 'Model')(args)
    model = nets.ModelWrapper(args, model)

    # forward here
    img_input = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3])
    image_ = img_input * 2.0 - 1.0
    image_ = tf.transpose(image_, [0, 3, 1, 2])
    lab_input = tf.placeholder(dtype=tf.int64, shape=[None])
    with TowerContext('', is_training=False):
        logits = model.get_logits(image_)

    # Session and load weights
    sess = tf.compat.v1.Session()
    get_model_loader(args.load).init(sess)

    model_adapted = utils_tf.ModelAdapter(logits, img_input, lab_input, sess, num_classes=1000)

    # load dataset

    dataset = ImageNetDataset(os.path.join(args.data, 'val'))
    chunkloader = ChunkDataset(dataset, num_chunk=args.actual_chunk,
                               total_chunks=args.total_chunks)
    dataloader = DataLoader(chunkloader, args.batch)

    print('\n\n\nTOTAL IMAGES ON THIS CHUNK: {}'.format(len(chunkloader)))

    # Count variables
    n = 0
    correct_dict = {'clean': 0}

    if args.save_adv:
        save_dict = {'gt': [],
                     'clean pred': [],
                     'adv pred': {},
                     'adv examples': {}}

    for numpy_clean, label in tqdm.tqdm(dataloader):

        if numpy_clean.shape[0] == 0:
            continue

        numpy_clean = numpy_clean.astype('float32') / 255.0

        # compute clean acc
        features = sess.run(logits, feed_dict={img_input: numpy_clean})
        clean_correct, clean_pred = compute_corrects(features, label)


        # compute adversarial images
        torch_clean = torch.from_numpy(np.transpose(numpy_clean, [0, 3, 1, 2])).to(dtype=torch.float)
        adversary = AutoAttack(model_adapted, norm='Linf', eps=args.epsilon,
                               version='standard', is_tf_model=True, verbose=True)
        try:
            adv_x = adversary.run_standard_evaluation_individual(torch_clean.contiguous(), torch.from_numpy(label), bs=numpy_clean.shape[0])
        except:
            print('SKIPPING STEP. ERROR FOUND ON AUTOATTACK')
            continue

        # compute robust acc
        adv_dict_correct = compute_corrects_advs(adv_x, label, sess, img_input, logits)

        print(adv_dict_correct)

        if n == 0:
            for k in adv_dict_correct.keys():
                correct_dict[k] = 0
                if args.save_adv:
                    if k == 'rob acc':
                        continue
                    save_dict['adv pred'][k] = []
                    save_dict['adv examples'][k] = []

        # update records
        n += label.shape[0]
        correct_dict['clean'] += clean_correct
        for k, v in adv_dict_correct.items():
            if k == 'rob acc':
                correct_dict[k] += v
            else:
                correct_dict[k] += v['corrects']

        if args.save_adv:
            save_dict['gt'].append(label)
            save_dict['clean pred'].append(clean_pred)
            for k, v in adv_dict_correct.items():
                if k == 'rob acc':
                    continue
                save_dict['adv pred'][k].append(v['pred'])
                save_dict['adv examples'][k].append(adv_x[k])

    message = f'n:{n}\n'
    for k, v in correct_dict.items():
        message += f'{k}:{v}\n'

    with open(f'{args.output_path}/results-chunk{args.actual_chunk}-total-chunks-{args.total_chunks}.txt', 'w') as f:
        f.write(message)
    print(message)

    if args.save_adv:
        save_dict['gt'] = np.concatenate(save_dict['gt'], axis=0)
        save_dict['clean pred'] = np.concatenate(save_dict['clean pred'], axis=0)
        for k in save_dict['adv pred'].keys():
            save_dict['adv pred'][k] = np.concatenate(save_dict['adv pred'][k], axis=0)
            save_dict['adv examples'][k] = np.concatenate(save_dict['adv examples'][k], axis=0)
        torch.save(save_dict, f'{args.output_path}/data-chunk{args.actual_chunk}-total-chunks-{args.total_chunks}.pth')


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

    # Initialize results
    results = {}
    with open(existing_files[0], 'r') as f:
        data = f.read()

    for line in data.split('\n'):
        if line == '':
            continue
        k, _ = line.split(':')
        results[k] = 0
        
    # read files
    for file in existing_files:
        with open(file, 'r') as f:
            data = f.read()

        for line in data.split('\n'):
            if line == '':
                continue
            k, v = line.split(':')
            v = int(v)
            results[k] += v

    message = 'Results:'
    for k, v in results.items():
        if k == 'n':
            message += '\n\t{} - {}'.format(k, v)
        else:
            message += '\n\t{} - {}%'.format(k, 100 * v / results['n'])

    print(message)
    with open(f'{args.output_path}/final-results.txt', 'w') as f:
        f.write(message)

if __name__ == '__main__':
    
    args = arguments()

    if args.evaluate_chunks:
        evaluate_chunks(args)
    else:
        main(args)
