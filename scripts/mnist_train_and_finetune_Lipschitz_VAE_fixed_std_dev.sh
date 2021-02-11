gammas=( 0.06324555 0.12649111 0.18973666 0.25298221 0.31622777 )
l_constant=10.0
latent_dim=10
saving_tag="continuous_bernoulli_std_dev_better_convergence_fixed_std_dev_gamma"
for gamma in "${gammas[@]}"
do
	echo "Training model for Lipschitz constant $l_constant..."
	pythonw ./lnets/tasks/vae/mains/train_VAE.py ./lnets/tasks/vae/configs/mnist/fc_VAE_bjorck_fixed_std_dev.json -o model.linear.bjorck_iter=3 model.encoder_mean.l_constant=$l_constant model.encoder_std_dev.l_constant=$l_constant model.decoder.l_constant=$l_constant model.encoder_std_dev.gamma=$gamma --saving_tag=$saving_tag
	echo "Finetuning model for Lipschitz constant $l_constant..."
	if [ "$saving_tag" == "" ]; then
		pythonw ./lnets/tasks/vae/mains/ortho_finetune.py --model.exp_path=./out/vae/mnist/not_finetuned/lipschitz_${l_constant}_latent_dim_${latent_dim}_${gamma}
	else 
		pythonw ./lnets/tasks/vae/mains/ortho_finetune.py --model.exp_path=./out/vae/mnist/not_finetuned/lipschitz_${l_constant}_latent_dim_${latent_dim}_${gamma}_${saving_tag} --saving_tag=${saving_tag}
	fi
done

