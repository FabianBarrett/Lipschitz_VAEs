# BB: Written starting August 14

# Implements latent space attacks for qualitative evaluation of VAE robustness

import argparse
import os
from munch import Munch
import json

import torch
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import os

from lnets.models import get_model
from lnets.data.load_data import load_data
from lnets.models.utils.conversion import convert_VAE_from_bjorck
from lnets.tasks.vae.mains.utils import orthonormalize_model, fix_groupings, get_target_image, sample_d_ball

# BB: Taken but adapted from Alex Camuto and Matthew Willetts
def latent_space_optimize_noise(model, config, image, target_image, initial_noise, soft=False, regularization_coefficient=None, maximum_noise_norm=None):

    adversarial_losses = []

    def fmin_func(noise):

        # BB: Soft determines whether latent space attack objective should be regularization_coefficient * norm of noise
        if soft:
          loss, gradient = model.eval_latent_space_attack(image, target_image, noise, soft=soft, regularization_coefficient=regularization_coefficient)
        # BB: If not, use hard constraint on norm of noise (i.e. attack is limited to this norm)
        else:
            loss, gradient = model.eval_latent_space_attack(image, target_image, noise, soft=soft, maximum_noise_norm=maximum_noise_norm)
        adversarial_losses.append(loss)
        return float(loss.data.numpy()), gradient.data.numpy().flatten().astype(np.float64)

    # BB: Bounds on the noise to ensure pixel values remain in interval [0, 1]
    lower_limit = -image.data.numpy().flatten()
    upper_limit = (1.0 - image.data.numpy().flatten())

    bounds = zip(lower_limit, upper_limit)
    bounds = [sorted(y) for y in bounds]

    # BB: Optimizer to find adversarial noise
    noise, _, _ = scipy.optimize.fmin_l_bfgs_b(fmin_func,
                                               x0=initial_noise,
                                               bounds=bounds,
                                               m=25,
                                               factr=10)
    return (torch.tensor(noise).view(1, 1, config.data.im_height, config.data.im_width)).float(), adversarial_losses

def get_attack_images(model, config, original_image, target_image, initial_noise, soft=False, regularization_coefficient=None, maximum_noise_norm=None):

    _, clean_reconstruction = model.loss(original_image)
    reshaped_clean_reconstruction = clean_reconstruction.view(1, 1, config.data.im_height, config.data.im_width)
    noise, _ = latent_space_optimize_noise(model, config, original_image, target_image, initial_noise, soft=soft, regularization_coefficient=regularization_coefficient, maximum_noise_norm=maximum_noise_norm)
    if not soft:
        noise = (maximum_noise_norm * noise.div(noise.norm(p=2))) if (noise.norm(p=2) > maximum_noise_norm) else noise
    
    noisy_image = original_image + noise.view(1, config.data.im_height, config.data.im_width)
    _, noisy_reconstruction = model.loss(noisy_image)
    reshaped_noisy_reconstruction = noisy_reconstruction.view(1, 1, config.data.im_height, config.data.im_width)
    image_compilation = torch.cat((original_image.unsqueeze(0), 
                                   reshaped_clean_reconstruction, 
                                   noise, 
                                   noisy_image.unsqueeze(0), 
                                   reshaped_noisy_reconstruction, 
                                   target_image.unsqueeze(0)), dim=-1)
    return image_compilation


def latent_space_attack(lipschitz_model, comparison_model, config, iterator, num_images, d_ball_init=True, soft=False, regularization_coefficient=None, maximum_noise_norm=None):

    sample = next(iter(iterator))
    attack_sample = (sample[0][:num_images], sample[1][:num_images])

    lipschitz_constant = config.model.encoder_mean.l_constant

    for index in range(num_images):
        print("Performing latent space attack {}...".format(index + 1))

        # Get original and target images
        original_image, original_class = attack_sample[0][index], attack_sample[1][index]
        target_image, target_class = get_target_image(attack_sample, original_class, index, num_images)

        if d_ball_init:
            # Sample initial noise for adversarial attack (same noise for both Lipschitz and comparison model)
            initial_noise = sample_d_ball(config.data.im_height * config.data.im_width, maximum_noise_norm).reshape((1, config.data.im_height, config.data.im_width)).astype(np.float32)

        else:
            # Sample initial noise for adversarial attack (same noise for both Lipschitz and comparison model)
            initial_noise = np.random.uniform(-1e-8, 1e-8, size=(1, config.data.im_height, config.data.im_width)).astype(np.float32)

        # Perform adversarial attack and get related images
        lipschitz_image_compilation = get_attack_images(lipschitz_model, config, original_image, target_image, initial_noise, soft=soft, regularization_coefficient=regularization_coefficient, maximum_noise_norm=maximum_noise_norm)
        comparison_image_compilation = get_attack_images(comparison_model, config, original_image, target_image, initial_noise, soft=soft, regularization_coefficient=regularization_coefficient, maximum_noise_norm=maximum_noise_norm)

        # Plotting
        plt.figure(figsize =(9, 3))
        plt.imshow(lipschitz_image_compilation.detach().squeeze().numpy())
        plt.axis('off')
        plotting_dir = "out/vae/attacks/latent_space_attacks/"
        if soft:
            plt.title("Latent space attack on VAE with Lipschitz constant: {}".format(lipschitz_constant) + "\n Regularization coefficient: {}".format(regularization_coefficient)) # + image_caption)
            plt.savefig(plotting_dir + "latent_attack_{}_soft_lipschitz_{}_reg_coefficient_{}.png".format(index + 1, lipschitz_constant, regularization_coefficient), dpi=300)
        else:
            plt.savefig(plotting_dir + "latent_attack_{}_hard_lipschitz_{}_maximum_perturbation_norm_{}.png".format(index + 1, lipschitz_constant, maximum_noise_norm), dpi=300)

        plt.figure(figsize = (9, 3))
        plt.imshow(comparison_image_compilation.detach().squeeze().numpy())
        plt.axis('off')
        plotting_dir = "out/vae/attacks/latent_space_attacks/"
        if soft:
            plt.title("Latent space attack on standard VAE" + "\n Regularization coefficient: {}".format(regularization_coefficient)) # + image_caption)
            plt.savefig(plotting_dir + "latent_attack_{}_soft_comparison_for_lipschitz_{}_reg_coefficient_{}.png".format(index + 1, lipschitz_constant, regularization_coefficient), dpi=300)
        else:
            plt.savefig(plotting_dir + "latent_attack_{}_hard_comparison_for_lipschitz_{}_maximum_perturbation_norm_{}.png".format(index + 1, lipschitz_constant, maximum_noise_norm), dpi=300)


def latent_attack_model(opt):

    lipschitz_model_exp_dir = opt['lipschitz_model']['exp_path']
    comparison_model_exp_dir = opt['comparison_model']['exp_path']

    lipschitz_model_path = os.path.join(lipschitz_model_exp_dir, 'checkpoints', 'best', 'best_model.pt')
    with open(os.path.join(lipschitz_model_exp_dir, 'logs', 'config.json'), 'r') as f:
        lipschitz_model_config = Munch.fromDict(json.load(f))

    comparison_model_path = os.path.join(comparison_model_exp_dir, 'checkpoints', 'best', 'best_model.pt')
    with open(os.path.join(comparison_model_exp_dir, 'logs', 'config.json'), 'r') as f:
        comparison_model_config = Munch.fromDict(json.load(f))

    lipschitz_model_config = fix_groupings(lipschitz_model_config)
    comparison_model_config = fix_groupings(comparison_model_config)

    bjorck_model = get_model(lipschitz_model_config)
    bjorck_model.load_state_dict(torch.load(lipschitz_model_path))

    comparison_model = get_model(comparison_model_config)
    comparison_model.load_state_dict(torch.load(comparison_model_path))

    if opt['data']['cuda']:
        print('Using CUDA')
        bjorck_model.cuda()
        comparison_model.cuda()

    lipschitz_model_config.data.cuda = opt['data']['cuda']
    data = load_data(lipschitz_model_config)

    # BB: Convert linear layers from Bjorck layers to standard linear layers
    standard_model = convert_VAE_from_bjorck(bjorck_model, lipschitz_model_config)

    # BB: Orthonormalize the final weight matrices
    orthonormalized_standard_model = orthonormalize_model(standard_model, lipschitz_model_config, iters=opt['ortho_iters'])

    orthonormalized_standard_model.eval()
    comparison_model.eval()

    if opt['soft']:
        print("Performing latent space attacks for Lipschitz constant {} with regularization coefficient {}...".format(lipschitz_model_config.model.encoder_mean.l_constant, opt['regularization_coefficient']))
    latent_space_attack(orthonormalized_standard_model, comparison_model, lipschitz_model_config, data['test'], opt['num_images'], d_ball_init=opt['d_ball_init'], soft=opt['soft'], regularization_coefficient=opt['regularization_coefficient'], maximum_noise_norm=opt['maximum_noise_norm'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Attack trained VAE')

    parser.add_argument('--lipschitz_model.exp_path', type=str, metavar='Lipschitz model path',
                    help="location of pretrained model weights to evaluate")
    parser.add_argument('--comparison_model.exp_path', type=str, metavar='Comparison model path', help="location of pretrained standard VAE weights")
    parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")
    parser.add_argument('--ortho_iters', type=int, default=50, help='number of orthonormalization iterations to run on standard linear layers')
    parser.add_argument('--num_images', type=int, default=10, help='number of images to perform latent space attack on')
    parser.add_argument('--soft', type=bool, default=False, help='whether latent attack should feature soft constraint on noise norm (hard constraint if False)')
    parser.add_argument('--d_ball_init', type=bool, default=True, help='whether attack noise should be initialized from random point in d-ball around image (True/False)')    
    parser.add_argument('--regularization_coefficient', type=float, default=1.0, help='regularization coefficient to use in latent space attack')
    parser.add_argument('--maximum_noise_norm', type=float, default=10.0, help='maximal norm of noise in max damage attack')

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

    latent_attack_model(opt)
