# BB: Written starting July 28

# Verifies the orthonormality of the encoder and decoder network weight matrices

import argparse
import os
from munch import Munch
import json

import torch

from lnets.models import get_model
from lnets.data.load_data import load_data
from lnets.models.utils.conversion import convert_VAE_from_bjorck
from lnets.tasks.vae.mains.utils import orthonormalize_model, \
                                        check_VAE_singular_values, \
                                        check_VAE_orthonormality, \
                                        visualize_reconstructions, \
                                        check_NLL, \
                                        fix_groupings

def check_model(opt):

    exp_dir = opt['model']['exp_path']

    model_path = os.path.join(exp_dir, 'checkpoints', 'best', 'best_model.pt')
    with open(os.path.join(exp_dir, 'logs', 'config.json'), 'r') as f:
        model_config = Munch.fromDict(json.load(f))

    model_config = fix_groupings(model_config)

    bjorck_model = get_model(model_config)
    bjorck_model.load_state_dict(torch.load(model_path))

    bjorck_model.eval()

    if opt['data']['cuda']:
        print('Using CUDA')
        model.cuda()

    model_config.data.cuda = opt['data']['cuda']
    data = load_data(model_config)

    # BB: Assumes encoder and decoder have same Lipschitz constants
    l_constant = model_config.model.encoder_mean.l_constant

    if opt['visualize']:
        # BB: Visualize reconstructions prior to model conversion, assumes encoder and decoder have same Lipschitz constants
        visualize_reconstructions(bjorck_model, data['test'], model_config, title_string='Bjorck linear layers \n Lipschitz constant {}'.format(l_constant))

    # BB: Convert linear layers from Bjorck layers to standard linear layers
    standard_model = convert_VAE_from_bjorck(bjorck_model, model_config)

    # BB: Orthonormalize the final weight matrices
    orthonormalized_standard_model = orthonormalize_model(standard_model, model_config, iters=opt['ortho_iters'])

    orthonormalized_standard_model.eval()

    if opt['visualize']:
        # BB: Visualize reconstructions after conversion and final orthonormalization, assumes encoder and decoder have same Lipschitz constants
        visualize_reconstructions(orthonormalized_standard_model, data['test'], model_config, title_string='Orthonormalized standard linear layers ({} iters) \n Lipschitz constant {}'.format(opt['ortho_iters'], l_constant))

    # BB: Inspect singular values of weight matrices after final orthonormalization
    check_VAE_singular_values(orthonormalized_standard_model)

    # BB: Check orthonormality of weight matrices after final orthonormalization
    check_VAE_orthonormality(orthonormalized_standard_model)

    # BB: Check negative log likelihood is on order of what would be expected from well-trained VAE
    check_NLL(orthonormalized_standard_model, data['test'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inspect properties of trained VAE')

    parser.add_argument('--model.exp_path', type=str, metavar='MODELPATH',
                    help="location of pretrained model weights to evaluate")
    parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")
    parser.add_argument('--visualize', type=bool, default=False, help="whether to visualize sample reconstructions (default: False)")
    parser.add_argument('--ortho_iters', type=int, default=50, help='number of orthonormalization iterations to run on standard linear layers')

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

    check_model(opt)