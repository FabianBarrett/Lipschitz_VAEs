l_constant=5.0
saving_tag="continuous_bernoulli_st_dev"
pythonw ./lnets/tasks/vae/mains/attack.py --lipschitz_model.exp_path=./out/vae/mnist/finetuned/lipschitz_${l_constant}_${saving_tag} --comparison_model.exp_path=./out/vae/mnist/not_finetuned/VAE_standard_continuous_bernoulli