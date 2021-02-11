import argparse
import os
from munch import Munch
import json

import torch
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker
from tqdm import tqdm
import seaborn as sns

from lnets.models import get_model
from lnets.data.load_data import load_data
from lnets.models.utils.conversion import convert_VAE_from_bjorck

def sigma_dependence(opt):

    models = []
    model_configs = []
    model_paths = []

    for model_dir in opt['model_dirs']:
        model_exp_dir = model_dir
        model_path = os.path.join(model_exp_dir, 'checkpoints', 'best', 'best_model.pt') 
        model_paths.append(model_path)
        with open(os.path.join(model_exp_dir, 'logs', 'config.json'), 'r') as f:
            model_config = Munch.fromDict(json.load(f))
            model_config = fix_groupings(model_config)
            model_configs.append(model_config)

    for index in range(len(model_configs)):
        model = get_model(model_configs[index])
        model.load_state_dict(torch.load(model_paths[index]))
        if opt['data']['cuda']:
            print('Using CUDA')
            model.cuda()
        models.append(model)

    for model_config in model_configs:
        model_config.model.encoder_std_dev = Munch(gamma=0.2)
        model_config.data.cuda = opt['data']['cuda']

    orthonormalized_models = []
    for index in range(len(models)):
        standard_model = convert_VAE_from_bjorck(models[index], model_configs[index])
        orthonormalized_model = orthonormalize_model(standard_model, model_configs[index], iters=opt['ortho_iters'])
        orthonormalized_model.eval()
        orthonormalized_model.gamma = 0.2
        orthonormalized_models.append(orthonormalized_model)

    torch.save(orthonormalized_models[0].state_dict(), os.getcwd() + "/temp/test.pt")
    model.load_state_dict(torch.load(os.getcwd() + "/temp/test.pt"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot margins')

    parser.add_argument('--model_dirs', type=str, nargs="+", help="locations of pretrained model weights to evaluate")
    parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")
    parser.add_argument('--ortho_iters', type=int, default=50, help='number of orthonormalization iterations to run on standard linear layers')
    parser.add_argument('--num_R_margin_images', type=int, default=10, help='number of images to estimate R margin for')
    parser.add_argument('--num_estimation_samples', type=int, default=20, help='number of forward passes to use for estimating r / capital R')
    parser.add_argument('--r', type=float, default=8.0, help='value of r to evaluate r-robustness probability for')
    parser.add_argument('--max_R', type=float, default=6.0, help='maximum value of R to test for in estimating r-robustness margin')
    parser.add_argument('--d_ball_init', type=bool, default=True, help='whether attack noise should be initialized from random point in d-ball around image (True/False)')
    parser.add_argument('--num_random_inits', type=int, default=10, help='how many random initializations of attack noise to use (int)')
    parser.add_argument('--margin_granularity', type=float, default=0.2, help='spacing between candidate R margins (smaller gives more exact estimate for more computation)')

    args = vars(parser.parse_args())

    print("Args: {}".format(args))

    opt = {}
    for k, v in args.items():
        cur = opt
        tokens = k.split('.')
        for token in tokens[:-1]:
            if token not in cur:
                cur[token] = {}
            cur = cur[token]
        cur[tokens[-1]] = v

    sigma_dependence(opt)