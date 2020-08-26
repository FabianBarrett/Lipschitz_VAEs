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

def visualize_latents(opt):

	lipschitz_exp_dir = opt['lipschitz_model']['exp_path']
	lipschitz_model_path = os.path.join(lipschitz_exp_dir, 'checkpoints', 'best', 'best_model.pt')
	with open(os.path.join(lipschitz_exp_dir, 'logs', 'config.json'), 'r') as f:
		lipschitz_model_config = Munch.fromDict(json.load(f))
		lipschitz_model_config = fix_groupings(lipschitz_model_config)
	lipschitz_model = get_model(lipschitz_model_config)
	lipschitz_model.load_state_dict(torch.load(lipschitz_model_path))

	standard_exp_dir = opt['standard_model']['exp_path']
	standard_model_path = os.path.join(standard_exp_dir, 'checkpoints', 'best', 'best_model.pt')
	with open(os.path.join(standard_exp_dir, 'logs', 'config.json'), 'r') as f:
		standard_model_config = Munch.fromDict(json.load(f))
	standard_model = get_model(standard_model_config)
	standard_model.load_state_dict(torch.load(standard_model_path))

	lipschitz_model = convert_VAE_from_bjorck(lipschitz_model, lipschitz_model_config)
	lipschitz_model = orthonormalize_model(lipschitz_model, lipschitz_model_config, iters=opt['ortho_iters'])

	data = load_data(lipschitz_model_config)

	sample_batch = next(iter(data['test']))

	lipschitz_model_latents = lipschitz_model.get_latents(sample_batch[0])
	standard_model_latents = standard_model.get_latents(sample_batch[0])

	colors = [color for color in mcolors.TABLEAU_COLORS][:len(sample_batch[1].unique())]
	plotting_dir = "out/vae/other_figures/latent_visualizations"
	x_bound = max(max(lipschitz_model_latents[:, 0].abs()), max(standard_model_latents[:, 0].abs())).detach().numpy()
	y_bound = max(max(lipschitz_model_latents[:, 1].abs()), max(standard_model_latents[:, 1].abs())).detach().numpy()
	unique_labels = sample_batch[1].unique()


	fig, ax = plt.subplots()
	for label_index in range(len(unique_labels)):
		indices = torch.nonzero(sample_batch[1] == unique_labels[label_index])
		ax.scatter(standard_model_latents[indices, 0].detach().numpy(), standard_model_latents[indices, 1].detach().numpy(), c=colors[label_index], label=unique_labels[label_index].item())
		ax.set_xlabel(r"$z_1$")
		ax.set_ylabel(r"$z_2$")
		ax.set_xlim(-x_bound, x_bound)
		ax.set_ylim(-y_bound, y_bound)
	ax.legend()
	fig.suptitle("Latent samples from a standard VAE")
	fig.savefig(plotting_dir + "/standard_VAE.png", dpi=300)
	fig.clf()

	fig, ax = plt.subplots()
	for label_index in range(len(unique_labels)):
		indices = torch.nonzero(sample_batch[1] == unique_labels[label_index])
		ax.scatter(lipschitz_model_latents[indices, 0].detach().numpy(), lipschitz_model_latents[indices, 1].detach().numpy(), c=colors[label_index], label=unique_labels[label_index].item())
		ax.set_xlabel(r"$z_1$")
		ax.set_ylabel(r"$z_2$")
		ax.set_xlim(-x_bound, x_bound)
		ax.set_ylim(-y_bound, y_bound)
	ax.legend()
	# Note: currently assumes Lipschitz constant of encoder mean, std dev and decoder are the same
	fig.suptitle("Latent samples from a VAE with Lipschitz constant {}".format(lipschitz_model_config.model.encoder_mean.l_constant))
	fig.savefig(plotting_dir + "/Lipschitz_VAE.png", dpi=300)
	fig.clf()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Attack trained VAE')
	parser.add_argument('--lipschitz_model.exp_path', type=str, metavar='Lipschitz model path', 
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
