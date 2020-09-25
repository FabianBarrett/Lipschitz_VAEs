l_constants=( 5.0 )
saving_tag="continuous_bernoulli_std_dev_better_convergence"
maximum_noise_norms=( 1.0 3.0 5.0 )
for l_constant in "${l_constants[@]}"
do
	for maximum_noise_norm in "${maximum_noise_norms[@]}"
	do
		pythonw ./lnets/tasks/vae/mains/latent_space_attack.py --lipschitz_model.exp_path=./out/vae/mnist/finetuned/lipschitz_${l_constant}_${saving_tag} --comparison_model.exp_path=./out/vae/mnist/not_finetuned/VAE_standard_continuous_bernoulli --maximum_noise_norm=${maximum_noise_norm}
	done
done