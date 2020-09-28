# Visualizes the latent spaces of VAEs with latent space \mathbb{R}^2

import argparse
import os
from munch import Munch
import json
import torch

from lnets.models import get_model
from lnets.data.load_data import load_data
from lnets.models.utils.conversion import convert_VAE_from_bjorck
from lnets.tasks.vae.mains.utils import orthonormalize_model, fix_groupings

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse

def visualize_latents(opt):

	lipschitz_exp_dir_1 = opt['lipschitz_model_1']['exp_path']
	lipschitz_model_path_1 = os.path.join(lipschitz_exp_dir_1, 'checkpoints', 'best', 'best_model.pt')
	with open(os.path.join(lipschitz_exp_dir_1, 'logs', 'config.json'), 'r') as f:
		lipschitz_model_config_1 = Munch.fromDict(json.load(f))
		lipschitz_model_config_1 = fix_groupings(lipschitz_model_config_1)
	lipschitz_model_1 = get_model(lipschitz_model_config_1)
	lipschitz_model_1.load_state_dict(torch.load(lipschitz_model_path_1))

	lipschitz_exp_dir_2 = opt['lipschitz_model_2']['exp_path']
	lipschitz_model_path_2 = os.path.join(lipschitz_exp_dir_2, 'checkpoints', 'best', 'best_model.pt')
	with open(os.path.join(lipschitz_exp_dir_2, 'logs', 'config.json'), 'r') as f:
		lipschitz_model_config_2 = Munch.fromDict(json.load(f))
		lipschitz_model_config_2 = fix_groupings(lipschitz_model_config_2)
	lipschitz_model_2 = get_model(lipschitz_model_config_2)
	lipschitz_model_2.load_state_dict(torch.load(lipschitz_model_path_2))

	standard_exp_dir = opt['standard_model']['exp_path']
	standard_model_path = os.path.join(standard_exp_dir, 'checkpoints', 'best', 'best_model.pt')
	with open(os.path.join(standard_exp_dir, 'logs', 'config.json'), 'r') as f:
		standard_model_config = Munch.fromDict(json.load(f))
	standard_model = get_model(standard_model_config)
	standard_model.load_state_dict(torch.load(standard_model_path))

	lipschitz_model_1 = convert_VAE_from_bjorck(lipschitz_model_1, lipschitz_model_config_1)
	lipschitz_model_1 = orthonormalize_model(lipschitz_model_1, lipschitz_model_config_1, iters=opt['ortho_iters'])

	lipschitz_model_2 = convert_VAE_from_bjorck(lipschitz_model_2, lipschitz_model_config_2)
	lipschitz_model_2 = orthonormalize_model(lipschitz_model_2, lipschitz_model_config_2, iters=opt['ortho_iters'])

	data = load_data(lipschitz_model_config_1)

	sample_batch = next(iter(data['test']))

	lipschitz_model_1_latents, lipschitz_encoder_mean_1, lipschitz_encoder_std_dev_1 = lipschitz_model_1.get_latents(sample_batch[0])
	lipschitz_model_2_latents, lipschitz_encoder_mean_2, lipschitz_encoder_std_dev_2 = lipschitz_model_2.get_latents(sample_batch[0])
	standard_model_latents, standard_encoder_mean, standard_encoder_std_dev = standard_model.get_latents(sample_batch[0])

	colors = [color for color in mcolors.TABLEAU_COLORS][:len(sample_batch[1].unique())]
	plotting_dir = "out/vae/other_figures/latent_visualizations"
	unique_labels = sample_batch[1].unique()

	# Visualize latents (mean + noise * std_dev)
	fig, ax = plt.subplots()
	for label_index in range(len(unique_labels)):
		indices = torch.nonzero(sample_batch[1] == unique_labels[label_index])
		ax.scatter(standard_model_latents[indices, 0].detach().numpy(), standard_model_latents[indices, 1].detach().numpy(), c=colors[label_index], label=unique_labels[label_index].item())
		ax.set_xlabel(r"$z_1$")
		ax.set_ylabel(r"$z_2$")
	ax.legend()
	fig.suptitle("Latent samples from a standard VAE")
	fig.savefig(plotting_dir + "/standard_VAE.png", dpi=300)
	fig.clf()

	fig, ax = plt.subplots()
	for label_index in range(len(unique_labels)):
		indices = torch.nonzero(sample_batch[1] == unique_labels[label_index])
		ax.scatter(lipschitz_model_1_latents[indices, 0].detach().numpy(), lipschitz_model_1_latents[indices, 1].detach().numpy(), c=colors[label_index], label=unique_labels[label_index].item())
		ax.set_xlabel(r"$z_1$")
		ax.set_ylabel(r"$z_2$")
	ax.legend()
	# Note: currently assumes Lipschitz constant of encoder mean, std dev and decoder are the same
	fig.suptitle("Latent samples from a VAE with Lipschitz constant {}".format(lipschitz_model_config_1.model.encoder_mean.l_constant))
	fig.savefig(plotting_dir + "/Lipschitz_VAE.png", dpi=300)
	fig.clf()

	# Visualize encoder means and standard deviations
	fig, ax = plt.subplots()
	for label_index in range(len(unique_labels)):
		indices = torch.nonzero(sample_batch[1] == unique_labels[label_index])
		ax.scatter(standard_encoder_mean[indices, 0].detach().numpy(), standard_encoder_mean[indices, 1].detach().numpy(), c=colors[label_index], label=unique_labels[label_index].item(), s=1)
		for index in range(len(indices)):
			ax.add_artist(Ellipse(xy=standard_encoder_mean[indices[index], :].squeeze(), width=standard_encoder_std_dev[indices[index], 0], height=standard_encoder_std_dev[indices[index], 1], color=colors[label_index], alpha=0.6))
		ax.set_xlabel(r"$z_1$")
		ax.set_ylabel(r"$z_2$")
		ax.set_xlim(-15, 15)
		ax.set_ylim(-15, 15)
	# Plot contours of prior (\mathcal{N}(\mathbf{0}, \mathbf{I})) for three standard deviations
	for standard_deviation in [1, 2, 3]:
		ax.add_artist(Ellipse(xy=[0.0, 0.0], width=standard_deviation * 1.0, height=standard_deviation * 1.0, edgecolor='black', facecolor='none', alpha=0.5, label=r"{}$\sigma$".format(int(standard_deviation))))
	ax.legend()
	fig.savefig(plotting_dir + "/standard_VAE_posterior.png", dpi=300)
	fig.clf()

	fig, ax = plt.subplots()
	for label_index in range(len(unique_labels)):
		indices = torch.nonzero(sample_batch[1] == unique_labels[label_index])
		ax.scatter(lipschitz_encoder_mean_1[indices, 0].detach().numpy(), lipschitz_encoder_mean_1[indices, 1].detach().numpy(), c=colors[label_index], label=unique_labels[label_index].item(), s=2)
		for index in range(len(indices)):
			ax.add_artist(Ellipse(xy=lipschitz_encoder_mean_1[indices[index], :].squeeze(), width=lipschitz_encoder_std_dev_1[indices[index], 0], height=lipschitz_encoder_std_dev_1[indices[index], 1], color=colors[label_index], alpha=0.6))
		ax.set_xlabel(r"$z_1$")
		ax.set_ylabel(r"$z_2$")
		ax.set_xlim(-15, 15)
		ax.set_ylim(-15, 15)
	# Plot contours of prior (\mathcal{N}(\mathbf{0}, \mathbf{I})) for three standard deviations
	for standard_deviation in [1, 2, 3]:
		ax.add_artist(Ellipse(xy=[0.0, 0.0], width=standard_deviation * 1.0, height=standard_deviation * 1.0, edgecolor='black', facecolor='none', alpha=0.4, label=r"{}$\sigma$".format(int(standard_deviation))))
	ax.legend()
	# Note: currently assumes Lipschitz constant of encoder mean, std dev and decoder are the same
	if 'beta' in opt['lipschitz_model_1']['exp_path']:
		fig.savefig(plotting_dir + "/Lipschitz_beta_VAE_1_posterior.png", dpi=300)
	else:
		fig.savefig(plotting_dir + "/Lipschitz_VAE_1_posterior.png", dpi=300)
	fig.clf()

	fig, ax = plt.subplots()
	for label_index in range(len(unique_labels)):
		indices = torch.nonzero(sample_batch[1] == unique_labels[label_index])
		ax.scatter(lipschitz_encoder_mean_2[indices, 0].detach().numpy(), lipschitz_encoder_mean_2[indices, 1].detach().numpy(), c=colors[label_index], label=unique_labels[label_index].item(), s=2)
		for index in range(len(indices)):
			ax.add_artist(Ellipse(xy=lipschitz_encoder_mean_2[indices[index], :].squeeze(), width=lipschitz_encoder_std_dev_2[indices[index], 0], height=lipschitz_encoder_std_dev_2[indices[index], 1], color=colors[label_index], alpha=0.6))
		ax.set_xlabel(r"$z_1$")
		ax.set_ylabel(r"$z_2$")
		ax.set_xlim(-15, 15)
		ax.set_ylim(-15, 15)
	# Plot contours of prior (\mathcal{N}(\mathbf{0}, \mathbf{I})) for three standard deviations
	for standard_deviation in [1, 2, 3]:
		ax.add_artist(Ellipse(xy=[0.0, 0.0], width=standard_deviation * 1.0, height=standard_deviation * 1.0, edgecolor='black', facecolor='none', alpha=0.4, label=r"{}$\sigma$".format(int(standard_deviation))))
	ax.legend()
	# Note: currently assumes Lipschitz constant of encoder mean, std dev and decoder are the same
	if 'beta' in opt['lipschitz_model_2']['exp_path']:
		fig.savefig(plotting_dir + "/Lipschitz_beta_VAE_2_posterior.png", dpi=300)
	else:
		fig.savefig(plotting_dir + "/Lipschitz_VAE_2_posterior.png", dpi=300)
	fig.clf()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Attack trained VAE')
	parser.add_argument('--lipschitz_model_1.exp_path', type=str, metavar='Lipschitz model 1 path', 
						help="location of pretrained Lipschitz model weights")
	parser.add_argument('--lipschitz_model_2.exp_path', type=str, metavar='Lipschitz model 2 path', 
						help="location of pretrained Lipschitz model weights")
	parser.add_argument('--standard_model.exp_path', type=str, metavar='Lipschitz model path', 
						help="location of pretrained Lipschitz model weights")
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

	visualize_latents(opt)
