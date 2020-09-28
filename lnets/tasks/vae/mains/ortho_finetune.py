# BB: Written starting July 15

"""
Do finetuning to ensure that the VAE's weight matrices are actually orthonormal.
"""

import json
import os.path
import argparse
from munch import Munch
import pprint

import torch

from lnets.models import get_model
from lnets.data.load_data import load_data
from lnets.models.layers import BjorckLinear
from lnets.tasks.vae.mains.train_VAE import train
from lnets.tasks.vae.mains.utils import visualize_reconstructions, fix_groupings

def main(opt):
    if not os.path.isdir(opt['output_root']):
        os.makedirs(opt['output_root'])

    exp_dir = opt['model']['exp_path']

    model_path = os.path.join(exp_dir, 'checkpoints', 'best', 'best_model.pt')
    with open(os.path.join(exp_dir, 'logs', 'config.json'), 'r') as f:
        model_config = Munch.fromDict(json.load(f))

    model_config = fix_groupings(model_config)

    model = get_model(model_config)
    model.load_state_dict(torch.load(model_path))

    if opt['data']['cuda']:
        print('Using CUDA')
        model.cuda()

    model_config.data.cuda = opt['data']['cuda']
    data = load_data(model_config)

    # Change the model to use ortho layers by copying the base weights
    bjorck_iters = opt['ortho_iters']

    for m in model.modules():
        if isinstance(m, BjorckLinear):
            m.config.model.linear.bjorck_iter = bjorck_iters

    model_config.output_root = os.path.join(opt['output_root'], model_config.data.name)

    model_config.optim.lr_schedule.lr_init = opt['finetuning_lr']

    if opt['max_grad_norm'] is not None:
        model_config.optim.max_grad_norm = opt['max_grad_norm']

    if not opt['to_convergence']:
        model_config.optim.to_convergence = opt['to_convergence']
        model_config.optim.epochs = opt['num_finetuning_epochs']

    if opt['convergence_tol'] is not None:
        model_config.optim.convergence_tol = opt['convergence_tol']

    model = train(model, data, model_config, finetune=True, saving_tag=opt['saving_tag'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do orthonormal finetuning on VAE')

    parser.add_argument('--model.exp_path', type=str, metavar='MODELPATH',
                        help="location of pretrained model weights to evaluate")
    parser.add_argument('--output_root', type=str, default="./out/vae/",
                        help='output directory to which results should be saved')
    parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")
    parser.add_argument('--ortho_iters', type=int, default=50, help='number of orthonormalization iterations to run on standard linear layers')
    parser.add_argument('--finetuning_lr', type=float, default=1e-5, help='learning rate for finetuning (default: 1e-5)')
    parser.add_argument('--saving_tag', type=str, default="", help='Note to add to output directory to distinguish between experiments')
    parser.add_argument('--max_grad_norm', type=float, default=None, help='maximum gradient norm during finetuning (should be smaller than 1e8; float)')
    parser.add_argument('--to_convergence', type=bool, default=True, help='whether to run finetuning until model converges (True/False)')
    parser.add_argument('--num_finetuning_epochs', type=int, default=5, help='if not running to convergence, number of finetuning epochs (int)')
    parser.add_argument('--convergence_tol', type=int, default=None, help='convergence criterion (difference between training losses across epochs; int)')

    args = vars(parser.parse_args())

    pp = pprint.PrettyPrinter()
    pp.pprint(args)

    opt = {}
    for k, v in args.items():
        cur = opt
        tokens = k.split('.')
        for token in tokens[:-1]:
            if token not in cur:
                cur[token] = {}
            cur = cur[token]
        cur[tokens[-1]] = v

    main(opt)