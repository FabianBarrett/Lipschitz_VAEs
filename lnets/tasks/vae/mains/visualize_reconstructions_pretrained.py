# BB: Written starting July 16

# BB: Run with e.g. python ./lnets/tasks/vae/mains/visualize_reconstructions_pretrained.py --model.exp_path=<trained_bjorck_model_path.pt> 

import argparse
import os
from munch import Munch
import json

import torch

from lnets.models import get_model
from lnets.data.load_data import load_data
from lnets.trainers.trainer import Trainer

import matplotlib.pyplot as plt

def visualize_reconstructions_pretrained(opt):

    exp_dir = opt['model']['exp_path']

    model_path = os.path.join(exp_dir, 'checkpoints', 'best', 'best_model.pt')
    with open(os.path.join(exp_dir, 'logs', 'config.json'), 'r') as f:
        model_config = Munch.fromDict(json.load(f))

        # Weird required hack to fix groupings (None is added to start during model training)
    if 'groupings' in model_config.model.encoder_mean and model_config.model.encoder_mean.groupings[0] is -1:
        model_config.model.encoder_mean.groupings = model_config.model.encoder_mean.groupings[1:]

    # Weird required hack to fix groupings (None is added to start during model training)
    if 'groupings' in model_config.model.encoder_st_dev and model_config.model.encoder_st_dev.groupings[0] is -1:
        model_config.model.encoder_st_dev.groupings = model_config.model.encoder_st_dev.groupings[1:]

    # Weird required hack to fix groupings (None is added to start during model training)
    if 'groupings' in model_config.model.decoder and model_config.model.decoder.groupings[0] is -1:
        model_config.model.decoder.groupings = model_config.model.decoder.groupings[1:]

    model = get_model(model_config)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    if opt['cuda']:
        print('Using CUDA')
        model.cuda()

    model_config.cuda = opt['cuda']

    data = load_data(model_config)

    sample = next(iter(data['test']))

    _, output = model.loss(sample, test=True)

    # Compare the original images and their reconstructions from a test batch
    original_images = sample[0].squeeze()
    reconstructions = output.view(sample[0].shape[0], model_config.data.im_height, -1)

    fig, ax = plt.subplots(opt['num_images'], 2, figsize=(6, opt['num_images']))
    for i in range(opt['num_images']):
        ax[i, 0].imshow(original_images[i, :].detach())
        ax[i, 0].axis('off')
        ax[i, 1].imshow(reconstructions[i, :].detach())
        ax[i, 1].axis('off')
    fig.suptitle("Original (left) and reconstructed (right) images")
    plt.show()
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize reconstructions of trained VAE')

    parser.add_argument('--model.exp_path', type=str, metavar='MODELPATH',
                    help="location of pretrained model weights to evaluate")
    parser.add_argument('--cuda', action='store_true', help="run in CUDA mode (default: False)")
    parser.add_argument('--num_images', type=int, default=10)

    args = vars(parser.parse_args())

    opt = {}
    for k, v in args.items():
        cur = opt
        tokens = k.split('.')
        for token in tokens[:-1]:
            if token not in cur:
                cur[token] = {}
            cur = cur[token]
        cur[tokens[-1]] = v

    visualize_reconstructions_pretrained(opt)
