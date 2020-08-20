import argparse
import os
from munch import Munch
import json

import torch
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from lnets.models import get_model
from lnets.data.load_data import load_data
from lnets.models.utils.conversion import convert_VAE_from_bjorck
from lnets.tasks.vae.mains.utils import orthonormalize_model, fix_groupings, sample_d_ball

# BB: Taken and modestly adapted from Alex Camuto and Matthew Willetts
def max_damage_optimize_noise(model, config, image, maximum_noise_norm, d_ball_init=True):

    if d_ball_init:
        initial_noise = sample_d_ball(config.data.im_height * config.data.im_width, maximum_noise_norm).reshape((1, config.data.im_height, config.data.im_width)).astype(np.float32)
    else:
        initial_noise = np.random.uniform(-1e-8, 1e-8, size=(1, config.data.im_height, config.data.im_width)).astype(np.float32)

    adversarial_losses = []

    def fmin_func(noise):

        loss, gradient = model.eval_max_damage_attack(image, noise, maximum_noise_norm)
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
                                                         m=100,
                                                         factr=10,
                                                         pgtol=1e-20)

    return (torch.tensor(noise).view(1, 1, config.data.im_height, config.data.im_width)).float(), adversarial_losses

# BB: Note the following is to be extended / built upon (i.e. further plots to come)
def get_max_damage_plot(models, model_configs, iterator, maximum_noise_norm, num_images, num_estimation_samples, r, d_ball_init=True):

    sample = next(iter(iterator))
    attack_sample = (sample[0][:num_images], sample[1][:num_images])

    # Find the probability of reconstructions within r under a max damage attack with given perturbation norm
    noise_norms = torch.linspace(1e-1, maximum_noise_norm, 10)
    
    print("Estimating r-robustness probability...")
    for image_index in range(num_images):
        print("Estimating probability for image {}...".format(image_index + 1))
        original_image = attack_sample[0][image_index]
        results = []
        for model_index in range(len(models)):
            print("Attacking image {} for model {}...".format(image_index + 1, model_index + 1))
            model = models[model_index]
            model_results = []
            for noise_norm in noise_norms:
                distances = []
                # BB: Return and check that it makes sense to be re-computing the adversarial noise
                for sample_index in tqdm(range(num_estimation_samples)):
                    _, clean_reconstruction = model.loss(original_image)
                    noise, _ = max_damage_optimize_noise(model, model_configs[0], original_image, noise_norm, d_ball_init=d_ball_init)
                    noise = (maximum_noise_norm * noise.div(noise.norm(p=2)))
                    noisy_image = original_image + noise.view(1, model_configs[0].data.im_height, model_configs[0].data.im_width)
                    _, noisy_reconstruction = model.loss(noisy_image)
                    distances.append((noisy_reconstruction.flatten() - clean_reconstruction.flatten()).norm(p=2))
                distances = torch.tensor(distances)
                estimated_probability = len(distances[distances <= r]) / num_estimation_samples
                model_results.append(estimated_probability)
            if model_configs[model_index].model.linear.type == "standard":
                results.append(("Standard VAE", model_results))
            else:
                # Note: This assumes the Lipschitz of the encoder and decoder are the same
                results.append(("Lipschitz constant " + str(model_configs[model_index].model.encoder_mean.l_constant), model_results))

        plt.clf()
        for model_results in results:
        	plt.plot(noise_norms.numpy(), np.array(model_results[1]), label=model_results[0])
        plt.legend()
        plt.xlabel(r"$|\delta_x|$")
        plt.ylabel(r"$\mathbb{P}(||\Delta||_2 \leq r)$")
        plt.ylim(bottom=0.0, top=1.2)
        plt.title("Max damage attacks on image {}".format(image_index + 1) + "\n (Estimated using {} samples for r={})".format(num_estimation_samples, r))
        plotting_dir = "out/vae/attacks/max_damage_attacks/"
        if d_ball_init:
            saving_string = "updated_r_robustness_probability_max_damage_example_{}_d_ball_init.png".format(image_index + 1)
        else:
            saving_string = "updated_r_robustness_probability_max_damage_example_{}_standard_init.png".format(image_index + 1)
        plt.savefig(plotting_dir + saving_string, dpi=300)
        plt.clf()

def max_damage_attack_model(opt):

    # Note: Assumes particular file naming convention (see training_getters in utils for reference)
    generic_exp_dir = opt['generic_model']['exp_path']

    model_configs = []
    model_paths = []
    for l_constant in opt['l_constants']:
        complete_exp_dir = generic_exp_dir.replace("+", str(l_constant))
        model_path = os.path.join(complete_exp_dir, 'checkpoints', 'best', 'best_model.pt')
        model_paths.append(model_path)
        with open(os.path.join(complete_exp_dir, 'logs', 'config.json'), 'r') as f:
            model_config = Munch.fromDict(json.load(f))
            model_config = fix_groupings(model_config)
            model_configs.append(model_config)

    models = []
    for index in range(len(model_configs)):
        model = get_model(model_configs[index])
        model.load_state_dict(torch.load(model_paths[index]))
        models.append(model)

    # Load comparison (standard VAE) model
    comparison_model_exp_dir = opt['comparison_model']['exp_path']
    comparison_model_path = os.path.join(comparison_model_exp_dir, 'checkpoints', 'best', 'best_model.pt')
    with open(os.path.join(comparison_model_exp_dir, 'logs', 'config.json'), 'r') as f:
        comparison_model_config = Munch.fromDict(json.load(f))

    comparison_model_config = fix_groupings(comparison_model_config)
    comparison_model = get_model(comparison_model_config)
    comparison_model.load_state_dict(torch.load(comparison_model_path))

    if opt['data']['cuda']:
        print('Using CUDA')
        for model in models:
            model.cuda()
        comparison_model.cuda()

    for model_config in model_configs:
        model_config.data.cuda = opt['data']['cuda']

    orthonormalized_models = []
    for index in range(len(models)):
    	standard_model = convert_VAE_from_bjorck(models[index], model_configs[index])
    	orthonormalized_models.append(orthonormalize_model(standard_model, model_configs[index], iters=opt['ortho_iters']))

    for orthonormalized_model in orthonormalized_models:
    	orthonormalized_model.eval()
    comparison_model.eval()

    data = load_data(model_configs[0])

    orthonormalized_models.append(comparison_model)
    model_configs.append(comparison_model_config)

    get_max_damage_plot(orthonormalized_models, model_configs, data['test'], opt['maximum_noise_norm'], opt['num_images'], opt['num_estimation_samples'], opt['r'], d_ball_init=opt['d_ball_init'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Attack trained VAE')

    parser.add_argument('--generic_model.exp_path', type=str, metavar='Lipschitz model path',
                    help="location of pretrained model weights to evaluate")
    parser.add_argument('--comparison_model.exp_path', type=str, metavar='Comparison model path', help="location of pretrained standard VAE weights")
    parser.add_argument('--l_constants', type=float, nargs="+", help="Lipschitz constants corresponding to models to attack")
    parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")
    parser.add_argument('--ortho_iters', type=int, default=50, help='number of orthonormalization iterations to run on standard linear layers')
    parser.add_argument('--num_images', type=int, default=3, help='number of images to perform latent space attack on')
    parser.add_argument('--num_estimation_samples', type=int, default=40, help='number of forward passes to use for estimating r / capital R')
    parser.add_argument('--r', type=float, default=10.0, help='value of r to evaluate r-robustness probability for')
    parser.add_argument('--maximum_noise_norm', type=float, default=10.0, help='maximal norm of noise in max damage attack')
    parser.add_argument('--d_ball_init', type=bool, default=True, help='whether attack noise should be initialized from random point in d-ball around image (True/False)')    

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

    max_damage_attack_model(opt)