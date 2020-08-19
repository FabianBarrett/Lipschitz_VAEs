l_constants=( 2.0 )
saving_tag="continuous_bernoulli_st_dev"
regularization_coefficients=( 1.0 )
for l_constant in "${l_constants[@]}"
do
	for regularization_coefficient in "${regularization_coefficients[@]}"
	do
		pythonw ./lnets/tasks/vae/mains/latent_space_attack.py --lipschitz_model.exp_path=./out/vae/mnist/finetuned/lipschitz_${l_constant}_${saving_tag} --comparison_model.exp_path=./out/vae/mnist/not_finetuned/VAE_standard_continuous_bernoulli --regularization_coefficient=${regularization_coefficient} --soft=True
	done
done