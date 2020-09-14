saving_tag="latent_dim_10_continuous_bernoulli_std_dev_better_convergence_constrained_std_dev"
pythonw ./lnets/tasks/vae/mains/max_damage_attack.py --generic_model.exp_path=./out/vae/mnist/finetuned/lipschitz_+_${saving_tag} --comparison_model.exp_path=./out/vae/mnist/not_finetuned/VAE_standard_continuous_bernoulli --certified=True --l_constants 10.0
