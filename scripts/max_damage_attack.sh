saving_tag="continuous_bernoulli_st_dev_better_convergence"
pythonw ./lnets/tasks/vae/mains/max_damage_attack.py --generic_model.exp_path=./out/vae/mnist/finetuned/lipschitz_+_${saving_tag} --comparison_model.exp_path=./out/vae/mnist/not_finetuned/VAE_standard_continuous_bernoulli --l_constants 5.0 10.0
