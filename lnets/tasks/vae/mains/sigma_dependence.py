# Roadmap for getting this plot
# 1. Build component in VAE model to set encoder standard dev as hyperparameter --> already set up, just need to get config file written... (and remind myself how to overwrite -- or maybe you can just rewrite the config itself...?)
# 2. Decide which sigma values to evaluate on
# 3. Figure out how many images to evaluate on (use several images to get semblance of error bars? or is that a bastardization? what's the randomness over? Which images were sampled I guess.)
# 4. Set up margin estimation to work...
# 5. Figure out whether to overlay empirical and theoretical margins or not...

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
from lnets.tasks.vae.mains.max_damage_attack import max_damage_optimize_noise, \
                                                    estimate_R_margin     
from lnets.tasks.vae.mains.utils import orthonormalize_model, \
                                        fix_groupings, \
                                        solve_bound_inequality, \
                                        process_bound_inequality_result, \
                                        solve_data_dep_bound_2


# Not necessary at the moment (hence commented out), but possibly revisit 

# def get_bound_margin(model_config, model, original_image, r, encoder_std_dev):
#     bound_inequality_result = solve_bound_inequality(model_config.model.decoder.l_constant, 
#                                                         model_config.model.encoder_mean.l_constant, 
#                                                         model_config.model.encoder_std_dev.l_constant, 
#                                                         r, encoder_std_dev.norm(p=2))
#     margin_1 = process_bound_inequality_result(bound_inequality_result)
#     margin_2 = solve_data_dep_bound_2(model_config.model.decoder.l_constant, model_config.model.encoder_mean.l_constant, model_config.model.encoder_std_dev.l_constant, r, model_config.model.latent_dim, encoder_std_dev.norm(p=2))
#     return max(margin_1, margin_2)

def get_gamma_from_sigma(m, sigma_norm):
    return sigma_norm / np.sqrt(m)

def get_sigma_from_gamma(m, gamma):
    return gamma * np.sqrt(m)

def evaluate_sigma_dependence(models, model_configs, iterator, num_images, max_R, num_estimation_samples, r, margin_granularity, num_random_inits, d_ball_init=True):

    sample = next(iter(iterator))
    attack_sample = (sample[0][:num_images], sample[1][:num_images])
    model_margins = []
    for model_index in range(len(models)):
        print("Estimating r-robustness margins for model {}...".format(model_index + 1))
        image_margins = []
        for image_index in tqdm(range(num_images)):
            original_image = attack_sample[0][image_index]
            model = models[model_index]
            estimated_margin = estimate_R_margin(model, model_configs[model_index], original_image, max_R, num_estimation_samples, r, margin_granularity, num_random_inits, d_ball_init=d_ball_init)
            image_margins.append(estimated_margin)
        # Note: This assumes the Lipschitz of the encoder and decoder are the same
        model_margins.append(((model_configs[model_index].model.encoder_std_dev.gamma), image_margins))

    # # # Currently assumes we are only evaluating a single model (possibly hard to plot otherwise, but can be revisited)
    # # model = models[0]
    # # # print(hasattr(model, "gamma"))
    # # # if hasattr(model, "gamma"):
    # # #     print(model.gamma)
    # # # raise RuntimeError
    # # model_config = model_configs[0]

    # sample = next(iter(iterator))
    # attack_sample = (sample[0][:num_images], sample[1][:num_images])

    # sigma_values = np.linspace(1e-2, 3, 5) ### CHANGE BACK TO 20/30 VALUES! ###

    # sigma_margins = []
    # for sigma_index in range(len(sigma_values)):
    #     list(model.children())[0].gamma = get_gamma_from_sigma(model_config.model.latent_dim, sigma_values[sigma_index])
    #     model_config.model.encoder_std_dev.gamma = get_gamma_from_sigma(model_config.model.latent_dim, sigma_values[sigma_index])
    #     # Hack to change encoder std dev (requires saving and re-loading model for each distinct value of ||\sigma||_2)
    #     # list(model.children())[0].gamma = 0.2
    #     # torch.save(orthonormalized_models[0].state_dict(), os.getcwd() + "/temp/test.pt")
    #     # orthonormalized_models[0] = get_model(fix_groupings(model_config))
    #     # orthonormalized_models[0].load_state_dict(torch.load(os.getcwd() + "/temp/test.pt"))
    #     image_margins = []
    #     for image_index in range(num_images):
    #         original_image = attack_sample[0][image_index]
    #         # Estimating $r$-robustness margin of model for image f{image_index} and f${sigma=}$.
    #         # estimated_margin = estimate_R_margin(model, model_config, original_image, max_R, num_estimation_samples, r, margin_granularity, num_random_inits, d_ball_init=d_ball_init, gamma=get_gamma_from_sigma(model_config.model.latent_dim, sigma_values[sigma_index]))
    #         estimated_margin = estimate_R_margin(model, model_config, original_image, max_R, num_estimation_samples, r, margin_granularity, num_random_inits, d_ball_init=d_ball_init)
    #         image_margins.append(estimated_margin)
    #     sigma_margins.append(image_margins)

    # print("Unsorted gammas: {}".format([model_configs[model_index].model.encoder_std_dev.gamma for model_index in range(len(models))]))

    means = np.array([np.mean(model_margins[model_index][1]) for model_index in range(len(models))])
    sds = np.array([np.std(model_margins[model_index][1]) for model_index in range(len(models))])
    gammas = [model_configs[model_index].model.encoder_std_dev.gamma for model_index in range(len(models))]
    sigmas = [get_sigma_from_gamma(model_configs[0].model.latent_dim, gamma) for gamma in gammas]

    sns.set(rc={"figure.figsize": (4, 4)}, style="whitegrid", font_scale=1.5)
    # plt.plot(sigma_values, sigma_means, label="Mean")
    # plt.fill_between(sigma_values, sigma_means - 2 * sigma_sds, sigma_means + 2 * sigma_sds, alpha=0.4)
    plt.errorbar(sigmas, means, yerr=sds, marker="o", linestyle='None')
    plt.ylabel(r"Estimated $R^r(x)$")
    plt.xlabel(r"$||\sigma||_2$")
    # plt.legend()
    plt.tight_layout()
    plotting_dir = "out/vae/attacks/R_margins/"
    saving_string = "sigma_dependence_R_margins_Lipschitz_{}.png".format(model_configs[0].model.encoder_mean.l_constant)
    plt.savefig(plotting_dir + saving_string, dpi=300)

    # model_margins = []
    # for model_index in range(len(models)):
    #     print("Estimating r-robustness margins for model {}...".format(model_index + 1))
    #     image_margins = []
    #     bound_margins = []
    #     for image_index in tqdm(range(num_images)):
    #         original_image = attack_sample[0][image_index]
    #         model = models[model_index]
    #         estimated_margin = estimate_R_margin(model, model_configs[model_index], original_image, max_R, num_estimation_samples, r, margin_granularity, num_random_inits, d_ball_init=d_ball_init)
    #         image_margins.append(estimated_margin)
    #         encoder_std_dev = model.loss(original_image, get_encoder_std_dev=True)
    #         bound_margin = get_bound_margin(model_configs[model_index], models[model_index], original_image, r, encoder_std_dev.detach())
    #         bound_margins.append(bound_margin)
    #     # Note: This assumes the Lipschitz of the encoder and decoder are the same
    #     model_margins.append(("Lipschitz constant " + str(model_configs[model_index].model.encoder_mean.l_constant), image_margins, bound_margins))

    # colors = [color for color in mcolors.TABLEAU_COLORS][:len(models)]

    # # Plot estimated R margins against margins implied by bounds
    # # plt.figure(figsize=(5, 5))
    # sns.set(rc={"figure.figsize": (4, 4)}, style="whitegrid", font_scale=1.5)
    # for model_index in range(len(models)):
    #     plt.plot(np.array(model_margins[model_index][2]), np.array(model_margins[model_index][1]), color=colors[0], linestyle='None', marker='o', fillstyle='full')
    #     print("Estimated R margin for VAE with Lipschitz constant {}: {}".format(str(model_configs[model_index].model.encoder_mean.l_constant), model_margins[model_index][1]))
    #     print("Theoretical margin for VAE with Lipschitz constant {}: {}".format(str(model_configs[model_index].model.encoder_mean.l_constant), model_margins[model_index][2]))
    # # max_value = (-1)
    # # for model_index in range(len(models)):
    # #     if max(max(model_margins[model_index][1]), max(model_margins[model_index][2])) > max_value:
    # #         max_value = max(max(model_margins[model_index][1]), max(model_margins[model_index][2]))
    # plt.plot(np.linspace(0, 2), np.linspace(0, 2), color='black', label='y=x')
    # # plt.plot([0.5, 0.75], [0.5, 0.75], color='black', label='y=x')
    # plt.xlabel(r"$R^r(x)$ bounds (log scale)")
    # plt.ylabel(r"Estimated $R^r(x)$")
    # plt.xscale('log')
    # plt.ylim(-0.3, 6.1)
    # # plt.title(r"Estimated $R^r(x)$ vs. $R^r(x)$ bounds for" + " r={}".format(r))
    # plt.legend()
    # plt.tight_layout()
    # plotting_dir = "out/vae/attacks/R_margins/"
    # # saving_string = "revised_estimated_vs_theoretical_R_margins_Lipschitz_{}.png".format(str(model_configs[model_index].model.encoder_mean.l_constant))
    # saving_string = "fashion_mnist_estimated_vs_theoretical_R_margins_Lipschitz.png"
    # plt.savefig(plotting_dir + saving_string, dpi=300)

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
        model_config.data.cuda = opt['data']['cuda']

    orthonormalized_models = []
    for index in range(len(models)):
        standard_model = convert_VAE_from_bjorck(models[index], model_configs[index])
        orthonormalized_model = orthonormalize_model(standard_model, model_configs[index], iters=opt['ortho_iters'])
        orthonormalized_model.eval()
        orthonormalized_models.append(orthonormalized_model)

    data = load_data(model_configs[0])

    # Plot empirical estimates of margin against hyperparameter set for encoder standard deviation
    evaluate_sigma_dependence(orthonormalized_models, model_configs, data['test'], opt['num_R_margin_images'], opt['max_R'], opt['num_estimation_samples'], opt['r'], opt['margin_granularity'], opt['num_random_inits'], d_ball_init=opt['d_ball_init'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot margins')

    parser.add_argument('--model_dirs', type=str, nargs="+", help="locations of pretrained model weights to evaluate")
    parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")
    parser.add_argument('--ortho_iters', type=int, default=50, help='number of orthonormalization iterations to run on standard linear layers')
    parser.add_argument('--num_R_margin_images', type=int, default=25, help='number of images to estimate R margin for') ### CHANGE THIS BACK TO 25 ###
    parser.add_argument('--num_estimation_samples', type=int, default=20, help='number of forward passes to use for estimating r / capital R') ### CHANGE THIS BACK TO 20 FROM 1 ###
    parser.add_argument('--r', type=float, default=8.0, help='value of r to evaluate r-robustness probability for')
    parser.add_argument('--max_R', type=float, default=6.0, help='maximum value of R to test for in estimating r-robustness margin') ### EXPERIMENT WITH THIS ###
    parser.add_argument('--d_ball_init', type=bool, default=True, help='whether attack noise should be initialized from random point in d-ball around image (True/False)')
    parser.add_argument('--num_random_inits', type=int, default=10, help='how many random initializations of attack noise to use (int)') ### CHANGE THIS BACK TO 10 FROM 1 ###
    parser.add_argument('--margin_granularity', type=float, default=0.2, help='spacing between candidate R margins (smaller gives more exact estimate for more computation)') ### CHANGE THIS BACK TO 0.2 ###

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