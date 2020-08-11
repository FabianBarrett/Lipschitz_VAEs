# BB: Written starting July 28

# Run with e.g. pythonw ./lnets/tasks/vae/mains/model_checks.py --model.exp_path=./out/vae/binarized_mnist/finetuned/model_directory

import argparse
import os
from munch import Munch
import json

import torch

from lnets.models import get_model
from lnets.data.load_data import load_data
from lnets.trainers.trainer import Trainer
from lnets.models.utils.conversion import convert_VAE_from_bjorck
from lnets.tasks.vae.mains.utils import orthonormalize_model, \
                                        check_VAE_singular_values, \
                                        check_VAE_orthonormality, \
                                        visualize_reconstructions, \
                                        check_NLL #, \
                                        # get_theoretical_bound

def check_model(opt):

    exp_dir = opt['model']['exp_path']

    model_path = os.path.join(exp_dir, 'checkpoints', 'best', 'best_model.pt')
    with open(os.path.join(exp_dir, 'logs', 'config.json'), 'r') as f:
        model_config = Munch.fromDict(json.load(f))

    # Weird required hack to fix groupings (None is added to start during model training)
    if 'groupings' in model_config.model.encoder_mean and model_config.model.encoder_mean.groupings[0] is -1:
        model_config.model.encoder_mean.groupings = model_config.model.encoder_mean.groupings[1:]

    # Weird required hack to fix groupings (None is added to start during model training)
    if 'groupings' in model_config.model.encoder_variance and model_config.model.encoder_variance.groupings[0] is -1:
        model_config.model.encoder_variance.groupings = model_config.model.encoder_variance.groupings[1:]

    # Weird required hack to fix groupings (None is added to start during model training)
    if 'groupings' in model_config.model.decoder and model_config.model.decoder.groupings[0] is -1:
        model_config.model.decoder.groupings = model_config.model.decoder.groupings[1:]

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

    # standard_model.eval()

    # # BB: Visualize reconstructions prior to final orthonormalization
    # # BB: Note, these are poor because the model expects orthonormalization in the forward pass
    # visualize_reconstructions(standard_model, data['test'], model_config, title_string='Standard linear layers (not orthonormalized)')

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
    
    # # BB: Get data-independent lower bound on probability of reconstruction within r given perturbation of norm < perturbation_norm
    # theoretical_bound = get_theoretical_bound(model_config, opt['r'], opt['perturbation_norm'])
    # print("Lower bound on probability of reconstruction within radius={} of unperturbed reconstruction given perturbation of norm <{}: {}".format(opt['r'], opt['perturbation_norm'], theoretical_bound))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inspect properties of trained VAE')

    parser.add_argument('--model.exp_path', type=str, metavar='MODELPATH',
                    help="location of pretrained model weights to evaluate")
    parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")
    parser.add_argument('--visualize', type=bool, default=False, help="whether to visualize sample reconstructions (default: False)")
    parser.add_argument('--ortho_iters', type=int, default=50, help='number of orthonormalization iterations to run on standard linear layers')
    parser.add_argument('--r', type=float, default=5.0, help='desired radius between reconstructions')
    parser.add_argument('--perturbation_norm', type=float, default=5.0, help='maximum perturbation in input space')

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