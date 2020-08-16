# BB: Written starting August 14

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
from lnets.tasks.vae.mains.utils import orthonormalize_model

def get_target_image(batch, input_class, input_index, num_images):
    image_counter = 0
    image_index = input_index + 1
    while image_counter < (num_images - 1):
        image_index %= num_images
        if batch[1][image_index] != input_class:
            return batch[0][image_index], batch[1][image_index]
        image_counter += 1
        image_index += 1
    raise RuntimeError("No appropriate target image found.")

def latent_space_attack(model, config, iterator, num_images, regularization_coefficient=1.0):

    sample = next(iter(iterator))
    attack_sample = (sample[0][:num_images], sample[1][:num_images])

    for index in range(num_images):
        print("Performing latent space attack {}...".format(index + 1))
        original_image, original_class = attack_sample[0][index], attack_sample[1][index]
        target_image, target_class = get_target_image(attack_sample, original_class, index, num_images)
        perturbed_image, adversarial_losses = latent_space_optimize_noise(model, config, original_image, target_image, regularization_coefficient)
        _, perturbed_reconstruction = model.loss(perturbed_image)
        _, unperturbed_reconstruction = model.loss(original_image)
        reshaped_perturbed_reconstruction = perturbed_reconstruction.view(1, 1, config.data.im_height, config.data.im_width)
        reshaped_unperturbed_reconstruction = unperturbed_reconstruction.view(1, 1, config.data.im_height, config.data.im_width)

        original_perturbed_target_reconstruction = torch.cat((original_image.unsqueeze(0), reshaped_unperturbed_reconstruction, perturbed_image, reshaped_perturbed_reconstruction, target_image.unsqueeze(0)), dim=-1)
        plt.imshow(original_perturbed_target_reconstruction.detach().squeeze().numpy())
        plt.axis('off')
        plt.title("Left to right: Original image, original reconstruction, \n perturbed image, perturbed reconstruction, target")
        plotting_dir = "out/vae/attacks/latent_space_attacks/"
        plt.savefig(plotting_dir + "latent_attack_{}".format(index + 1), dpi=300)

# BB: Note I haven't checked the internals of this thoroughly
# BB: Taken but adapted from Alex Camuto and Matthew Willetts
def max_damage_optimize_noise(model, config, image, maximum_noise_norm):

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
	perturbed_image, _, _ = scipy.optimize.fmin_l_bfgs_b(fmin_func,
                                                                  x0=initial_noise,
                                                                  bounds=bounds,
                                                                  m=100,
                                                                  factr=10,
                                                                  pgtol=1e-20)
	return (torch.tensor(perturbed_image).view(1, 1, config.data.im_height, config.data.im_width)).float(), adversarial_losses

def latent_space_optimize_noise(model, config, image, target_image, regularization_coefficient):

	initial_noise = np.random.uniform(-1e-8, 1e-8, size=(1, config.data.im_height, config.data.im_width)).astype(np.float32)

	adversarial_losses = []

	def fmin_func(noise):

		loss, gradient = model.eval_latent_space_attack(image, target_image, noise, regularization_coefficient)
		adversarial_losses.append(loss)
		return float(loss.data.numpy()), gradient.data.numpy().flatten().astype(np.float64)

	# BB: Bounds on the noise to ensure pixel values remain in interval [0, 1]
	lower_limit = -image.data.numpy().flatten()
	upper_limit = (1.0 - image.data.numpy().flatten())

	bounds = zip(lower_limit, upper_limit)
	bounds = [sorted(y) for y in bounds]

	# BB: Optimizer to find adversarial noise
	perturbed_image, _, _ = scipy.optimize.fmin_l_bfgs_b(fmin_func,
                                                                  x0=initial_noise,
                                                                  bounds=bounds,
                                                                  m=25,
                                                                  factr=10)
	return (torch.tensor(perturbed_image).view(1, 1, config.data.im_height, config.data.im_width)).float(), adversarial_losses

def attack_model(opt):

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

    bjorck_model = get_model(model_config)
    bjorck_model.load_state_dict(torch.load(model_path))

    bjorck_model.eval()

    if opt['data']['cuda']:
        print('Using CUDA')
        model.cuda()

    model_config.data.cuda = opt['data']['cuda']
    data = load_data(model_config)

    # BB: Convert linear layers from Bjorck layers to standard linear layers
    standard_model = convert_VAE_from_bjorck(bjorck_model, model_config)

    # BB: Orthonormalize the final weight matrices
    orthonormalized_standard_model = orthonormalize_model(standard_model, model_config, iters=opt['ortho_iters'])

    orthonormalized_standard_model.eval()

    print("Performing latent space attacks...")
    latent_space_attack(orthonormalized_standard_model, model_config, data['test'], opt['num_images'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Attack trained VAE')

    parser.add_argument('--model.exp_path', type=str, metavar='MODELPATH',
                    help="location of pretrained model weights to evaluate")
    parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")
    parser.add_argument('--ortho_iters', type=int, default=50, help='number of orthonormalization iterations to run on standard linear layers')
    parser.add_argument('--num_images', type=int, default=10, help='number of images to perform latent space attack on')

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

    attack_model(opt)
