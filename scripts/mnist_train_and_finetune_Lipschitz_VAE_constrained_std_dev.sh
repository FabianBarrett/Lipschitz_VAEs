l_constants=( 6.0 8.0 9.0 )
latent_dim=10
saving_tag="continuous_bernoulli_std_dev_better_convergence_constrained_std_dev"
for l_constant in "${l_constants[@]}"
do
	echo "Training model for Lipschitz constant $l_constant..."
	pythonw ./lnets/tasks/vae/mains/train_VAE.py ./lnets/tasks/vae/configs/mnist/fc_VAE_bjorck_constrained_std_dev.json -o model.linear.bjorck_iter=3 model.encoder_mean.l_constant=$l_constant model.encoder_std_dev.l_constant=$l_constant model.decoder.l_constant=$l_constant model.encoder_std_dev.std_dev_norm=0.1 --saving_tag=$saving_tag
	echo "Finetuning model for Lipschitz constant $l_constant..."
	if [ "$saving_tag" == "" ]; then
		pythonw ./lnets/tasks/vae/mains/ortho_finetune.py --model.exp_path=./out/vae/mnist/not_finetuned/lipschitz_${l_constant}_latent_dim_${latent_dim}
	else 
		pythonw ./lnets/tasks/vae/mains/ortho_finetune.py --model.exp_path=./out/vae/mnist/not_finetuned/lipschitz_${l_constant}_latent_dim_${latent_dim}_${saving_tag} --saving_tag=${saving_tag}
	fi
done