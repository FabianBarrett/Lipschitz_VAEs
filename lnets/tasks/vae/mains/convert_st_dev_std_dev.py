# Converts old models trained under one naming convention to the new naming convention

from munch import Munch
import json
import os
import torch
from collections import OrderedDict
from lnets.tasks.vae.mains.utils import fix_groupings
from lnets.models import get_model
from lnets.utils.saving_and_loading import save_model

if __name__ == '__main__':
	for lipschitz_constant in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
		for directory in ["finetuned", "not_finetuned"]:
			exp_dir = "./out/vae/mnist/{}/lipschitz_{}_continuous_bernoulli_std_dev_better_convergence".format(directory, lipschitz_constant)
			model_path = os.path.join(exp_dir, 'checkpoints', 'best', 'best_model.pt')
			model_as_pt = torch.load(model_path)
			with open(os.path.join(exp_dir, 'logs', 'config.json'), 'r') as f:
				model_config = Munch.fromDict(json.load(f))
				model_config = fix_groupings(model_config)
			new_model_as_pt = OrderedDict([])
			for old_key, value in model_as_pt.items():
				if "st_dev" in old_key:
					new_key = old_key.replace("st_dev", "std_dev")
					new_model_as_pt[new_key] = value
				else:
					new_model_as_pt[old_key] = value
			model = get_model(model_config)
			model.load_state_dict(new_model_as_pt)
			temp_saving_path = exp_dir + '/checkpoints/best/best_model.pt'
			save_model(model, temp_saving_path)

	standard_model_dir = "./out/vae/mnist/not_finetuned/VAE_standard_continuous_bernoulli"
	standard_model_path = os.path.join(standard_model_dir, 'checkpoints', 'best', 'best_model.pt')
	standard_model_as_pt = torch.load(standard_model_path)
	with open(os.path.join(standard_model_dir, 'logs', 'config.json'), 'r') as f:
		standard_model_config = Munch.fromDict(json.load(f))
		standard_model_config = fix_groupings(standard_model_config)
	standard_new_model_as_pt = OrderedDict([])
	for old_key, value in standard_model_as_pt.items():
		if "st_dev" in old_key:
			new_key = old_key.replace("st_dev", "std_dev")
			standard_new_model_as_pt[new_key] = value
		else:
			standard_new_model_as_pt[old_key] = value
	standard_model = get_model(standard_model_config)
	standard_model.load_state_dict(standard_new_model_as_pt)
	standard_temp_saving_path = standard_model_dir + '/checkpoints/best/best_model.pt'
	save_model(standard_model, standard_temp_saving_path)

