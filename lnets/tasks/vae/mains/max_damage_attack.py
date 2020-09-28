# Implements maximum damage attacks for quantitative evaluation of VAE robustness

import argparse
import os
from munch import Munch
import json

import torch
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

from lnets.models import get_model
from lnets.data.load_data import load_data
from lnets.models.utils.conversion import convert_VAE_from_bjorck
from lnets.tasks.vae.mains.utils import orthonormalize_model, \
                                        fix_groupings, \
                                        sample_d_ball, \
                                        solve_bound_inequality, \
                                        process_bound_inequality_result, \
                                        get_log_likelihood_Lipschitz_plot, \
                                        get_encoder_std_dev_Lipschitz_plot, \
                                        solve_bound_2

# BB: Taken and modestly adapted from Alex Camuto and Matthew Willetts
def max_damage_optimize_noise(model, config, image, maximum_noise_norm, d_ball_init=True, scale=False):

    if d_ball_init:
        initial_noise = sample_d_ball(config.data.im_height * config.data.im_width, maximum_noise_norm).reshape((1, config.data.im_height, config.data.im_width)).astype(np.float32)
    else:
        initial_noise = np.random.uniform(-1e-8, 1e-8, size=(1, config.data.im_height, config.data.im_width)).astype(np.float32)

    adversarial_losses = []

    def fmin_func(noise):

        loss, gradient = model.eval_max_damage_attack(image, noise, maximum_noise_norm, scale=scale)
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

def get_max_damage_plot(models, model_configs, iterator, maximum_noise_norm, num_images, num_estimation_samples, r, num_random_inits, d_ball_init=True):

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
                for random_init in tqdm(range(num_random_inits)):
                    noise, _ = max_damage_optimize_noise(model, model_configs[0], original_image, noise_norm, d_ball_init=d_ball_init, scale=False)
                    if noise.norm(p=2) > maximum_noise_norm:
                        noise = maximum_noise_norm * noise.div(noise.norm(p=2))
                    noisy_image = original_image + noise.view(1, model_configs[0].data.im_height, model_configs[0].data.im_width)
                    for sample_index in range(num_estimation_samples):
                        _, clean_reconstruction = model.loss(original_image)
                        _, noisy_reconstruction = model.loss(noisy_image)
                        distances.append((noisy_reconstruction.flatten() - clean_reconstruction.flatten()).norm(p=2))
                distances = torch.tensor(distances)
                estimated_probability = len(distances[distances <= r]) / (num_estimation_samples * num_random_inits)
                model_results.append(estimated_probability)
            if model_configs[model_index].model.linear.type == "standard":
                results.append(("Standard VAE", model_results))
            else:
                # Note: This assumes the Lipschitz of the encoder and decoder are the same
                results.append(("Lipschitz constant " + str(model_configs[model_index].model.encoder_mean.l_constant), model_results))

        colors = [color for color in mcolors.TABLEAU_COLORS][:len(models)]

        plt.clf()
        for model_results_index in range(len(results)):
        	plt.plot(noise_norms.numpy(), np.array(results[model_results_index][1]), label=results[model_results_index][0], color=colors[model_results_index])
        plt.legend()
        plt.xlabel(r"$|\delta_x|$")
        plt.ylabel(r"$\mathbb{P}(||\Delta||_2 \leq r)$")
        plt.ylim(bottom=0.0, top=1.2)
        plt.title("Max damage attacks on image {}".format(image_index + 1) + "\n (Estimated using {} samples for r={})".format(num_estimation_samples * num_random_inits, r))
        plotting_dir = "out/vae/attacks/max_damage_attacks/"
        if d_ball_init:
            saving_string = "updated_r_robustness_probability_max_damage_example_{}_d_ball_init.png".format(image_index + 1)
        else:
            saving_string = "updated_r_robustness_probability_max_damage_example_{}_standard_init.png".format(image_index + 1)
        plt.savefig(plotting_dir + saving_string, dpi=300)
        plt.clf()

# BB: Taken and modestly adapted from Alex Camuto and Matthew Willetts (we add random restarts)
def estimate_R_margin(model, config, image, max_R, num_estimation_samples, r, margin_granularity, num_random_inits, d_ball_init=True):
    candidate_margins = np.arange(1e-6, max_R, margin_granularity)
    estimated_probabilities = []
    for random_init in range(num_random_inits):
        distances = []
        noise, _ = max_damage_optimize_noise(model, config, image, candidate_margins[0], d_ball_init=d_ball_init, scale=True)
        noise = (candidate_margins[0] * noise.div(noise.norm(p=2)))
        noisy_image = image + noise.view(1, config.data.im_height, config.data.im_width)
        for _ in range(num_estimation_samples):
            _, clean_reconstruction = model.loss(image)
            _, noisy_reconstruction = model.loss(noisy_image)
            distances.append((noisy_reconstruction.flatten() - clean_reconstruction.flatten()).norm(p=2))
        distances = torch.tensor(distances)
        estimated_probability = len(distances[distances <= r]) / (num_estimation_samples)
        estimated_probabilities.append(estimated_probability)
    estimated_probabilities = torch.tensor(estimated_probabilities)
    if len(estimated_probabilities[estimated_probabilities <= 0.5]) >= 1:
        return candidate_margins[0]
    for candidate_margin in reversed(candidate_margins):
        estimated_probabilities = []
        for random_init in range(num_random_inits):
            distances = []
            noise, _ = max_damage_optimize_noise(model, config, image, candidate_margin, d_ball_init=d_ball_init, scale=True)
            noise = (candidate_margin * noise.div(noise.norm(p=2)))
            noisy_image = image + noise.view(1, config.data.im_height, config.data.im_width)
            for _ in range(num_estimation_samples):
                _, clean_reconstruction = model.loss(image)
                _, noisy_reconstruction = model.loss(noisy_image)
                distances.append((noisy_reconstruction.flatten() - clean_reconstruction.flatten()).norm(p=2))
            distances = torch.tensor(distances)
            estimated_probability = len(distances[distances <= r]) / (num_estimation_samples)
            estimated_probabilities.append(estimated_probability)
        estimated_probabilities = torch.tensor(estimated_probabilities)
        if len(estimated_probabilities[estimated_probabilities > 0.5]) == num_random_inits:
            return candidate_margin
    raise RuntimeError("Did not find R margin such that r-robustness was satisfied.")

def get_R_margins(models, model_configs, iterator, num_images, max_R, num_estimation_samples, r, margin_granularity, num_random_inits, d_ball_init=True, certified=False, fixed_std_dev=False):

    sample = next(iter(iterator))
    attack_sample = (sample[0][:num_images], sample[1][:num_images])

    model_margins = []
    print("Estimating r-robustness margins...")
    for model_index in range(len(models)):
        image_margins = []
        if model_configs[model_index].model.linear.type != "standard":
            bound_margins = []
        if model_configs[model_index].model.linear.type != "standard" and 'gamma' in model_configs[model_index].model.encoder_std_dev:
            bound_margin = solve_bound_2(model_configs[model_index].model.decoder.l_constant, 
                                             model_configs[model_index].model.encoder_mean.l_constant, 
                                             model_configs[model_index].model.encoder_std_dev.gamma, r,
                                             model_configs[model_index].model.latent_dim)
        for image_index in tqdm(range(num_images)):
            original_image = attack_sample[0][image_index]
            model = models[model_index]
            estimated_margin = estimate_R_margin(model, model_configs[0], original_image, max_R, num_estimation_samples, r, margin_granularity, num_random_inits, d_ball_init=d_ball_init)
            image_margins.append(estimated_margin)
            if model_configs[model_index].model.linear.type != "standard" and not fixed_std_dev:
                encoder_std_dev = models[model_index].loss(original_image, get_encoder_std_dev=True)
                bound_inequality_result = solve_bound_inequality(model_configs[model_index].model.decoder.l_constant, 
                                                                 model_configs[model_index].model.encoder_mean.l_constant, 
                                                                 model_configs[model_index].model.encoder_std_dev.l_constant, 
                                                                 r, encoder_std_dev.norm(p=2))
                bound_margins.append(process_bound_inequality_result(bound_inequality_result))
            if model_configs[model_index].model.linear.type != "standard" and 'gamma' in model_configs[model_index].model.encoder_std_dev:
                bound_margins.append(bound_margin)
        if model_configs[model_index].model.linear.type == "standard":
            model_margins.append(("Standard VAE", image_margins))
        else:
            # Note: This assumes the Lipschitz of the encoder and decoder are the same
            model_margins.append(("Lipschitz constant " + str(model_configs[model_index].model.encoder_mean.l_constant), image_margins, bound_margins))

    histogram_bins = np.arange(1e-6, max_R, margin_granularity)
    max_frequency = -1
    for margins in model_margins:
        if max_frequency < max(margins[1]):
            max_frequency = max(margins[1])

    colors = [color for color in mcolors.TABLEAU_COLORS][:len(models)]

    # Generate histograms of R margins w.r.t. model type
    plt.clf()
    fig, ax = plt.subplots(len(models), 1, figsize=(6, len(models) + 4))
    for model_index in range(len(models)):
        ax[model_index].hist(np.array(model_margins[model_index][1]), histogram_bins, label=model_margins[model_index][0], color=colors[model_index])
        ax[model_index].set_yticks([])
        ax[model_index].set_ylim([0, max_frequency + 2])
        ax[model_index].set_ylabel("Frequency")
        if model_index == (len(models) - 1):
            ax[model_index].set_xlabel(r"Estimated $R^r(x)$")
        else:
            ax[model_index].set_xticks([])
            ax[model_index].set_xlabel("")
        ax[model_index].legend()
    fig.suptitle(r"Estimated $R^r(x)$ for" + " r={}".format(r))
    plotting_dir = "out/vae/attacks/R_margins/"
    if d_ball_init:
        fig.savefig(plotting_dir + "estimated_R_margins_d_ball_init_certified_{}.png".format(str(certified).lower()), dpi=300)
    else:
        fig.savefig(plotting_dir + "estimated_R_margins_standard_init_certified_{}.png".format(str(certified).lower()), dpi=300)
    
    # Plot estimated R margins against margin implied by Markov bound
    for model_index in range(len(models)):
        if model_configs[model_index].model.linear.type != "standard":
            max_value = max(max(model_margins[model_index][2]), max(model_margins[model_index][1]))
            plt.clf()
            if not fixed_std_dev:
                plt.plot(np.array(model_margins[model_index][2]), np.array(model_margins[model_index][1]), color=colors[model_index], linestyle='None', marker='o', fillstyle='full')
            else:
                plt.plot(np.array(model_margins[model_index][2]), np.array(model_margins[model_index][1]), color=colors[1], linestyle='None', marker='o', fillstyle='full')
            print("Estimated R margin for VAE with Lipschitz constant {}: {}".format(str(model_configs[model_index].model.encoder_mean.l_constant), model_margins[model_index][1]))
            print("Theoretical R margin for VAE with Lipschitz constant {}: {}".format(str(model_configs[model_index].model.encoder_mean.l_constant), model_margins[model_index][2]))
            plt.plot(np.linspace(0, max_value), np.linspace(0, max_value), color='black')
            plt.xlabel(r"$R^r(x)$ bound (log scale)")
            plt.xscale('log')
            plt.ylabel(r"Estimated $R^r(x)$")
            plt.title(r"Estimated $R^r(x)$ vs. $R^r(x)$ bound for" + " r={}".format(r) + "\n Lipschitz constant: {}".format(str(model_configs[model_index].model.encoder_mean.l_constant)))
            if not fixed_std_dev:
                saving_name = "estimated_vs_theoretical_R_margins_d_ball_init_Lipschitz_{}_certified_{}.png".format(str(model_configs[model_index].model.encoder_mean.l_constant), str(certified).lower())
            else:
                saving_name = "estimated_vs_theoretical_R_margins_d_ball_init_Lipschitz_{}_fixed_std_dev.png".format(str(model_configs[model_index].model.encoder_mean.l_constant))
            plt.savefig(plotting_dir + saving_name, dpi=300)
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

    if not opt['certified']:
        # Note: The following two function calls are added to the file for convenience (since all models are pre-loaded), not because of conceptual similarity
        # Inspect the relationship between encoder standard deviation norm and encoder & decoder Lipschitz constant
        get_encoder_std_dev_Lipschitz_plot(orthonormalized_models, model_configs, data['test'])

        # Inspect the relationship between reconstruction quality and encoder & decoder Lipschitz constant
        get_log_likelihood_Lipschitz_plot(orthonormalized_models, model_configs, data['test'])

        # Inspect r-robustness probability degradation w.r.t. norm of max damage attacks and model
        get_max_damage_plot(orthonormalized_models, model_configs, data['test'], opt['maximum_noise_norm'], opt['num_max_damage_images'], opt['num_estimation_samples'], opt['r'], opt['num_random_inits'], d_ball_init=opt['d_ball_init'])

    # Inspect estimated R margin w.r.t. model
    get_R_margins(orthonormalized_models, model_configs, data['test'], opt['num_R_margin_images'], opt['max_R'], opt['num_estimation_samples'], opt['r'], opt['margin_granularity'], opt['num_random_inits'], d_ball_init=opt['d_ball_init'], certified=opt['certified'], fixed_std_dev=opt['fixed_std_dev'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Attack trained VAE')

    parser.add_argument('--generic_model.exp_path', type=str, metavar='Lipschitz model path',
                    help="location of pretrained model weights to evaluate")
    parser.add_argument('--comparison_model.exp_path', type=str, metavar='Comparison model path', help="location of pretrained standard VAE weights")
    parser.add_argument('--l_constants', type=float, nargs="+", help="Lipschitz constants corresponding to models to attack")
    parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")
    parser.add_argument('--ortho_iters', type=int, default=50, help='number of orthonormalization iterations to run on standard linear layers')
    parser.add_argument('--num_max_damage_images', type=int, default=3, help='number of images to perform attack on')
    parser.add_argument('--num_R_margin_images', type=int, default=25, help='number of images to estimate R margin for')
    parser.add_argument('--num_estimation_samples', type=int, default=40, help='number of forward passes to use for estimating r / capital R')
    parser.add_argument('--r', type=float, default=8.0, help='value of r to evaluate r-robustness probability for')
    parser.add_argument('--max_R', type=float, default=10.0, help='maximum value of R to test for in estimating r-robustness margin')
    parser.add_argument('--maximum_noise_norm', type=float, default=10.0, help='maximal norm of noise in max damage attack')
    parser.add_argument('--d_ball_init', type=bool, default=True, help='whether attack noise should be initialized from random point in d-ball around image (True/False)')
    parser.add_argument('--num_random_inits', type=int, default=5, help='how many random initializations of attack noise to use (int)')
    parser.add_argument('--margin_granularity', type=float, default=0.5, help='spacing between candidate R margins (smaller gives more exact estimate for more computation)')
    parser.add_argument('--certified', type=bool, default=False, help='flag to indicate whether model being evaluated should be certifiably robust')
    parser.add_argument('--fixed_std_dev', type=bool, default=False, help='flag to indicate whether VAE was trained with fixed encoder standard deviation')

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