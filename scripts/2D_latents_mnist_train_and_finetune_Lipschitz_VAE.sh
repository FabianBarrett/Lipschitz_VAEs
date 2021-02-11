l_constants=( 20.0 )
saving_tag="continuous_bernoulli_std_dev_better_convergence"
for l_constant in "${l_constants[@]}"
do
	echo "Training model for Lipschitz constant $l_constant..."
	pythonw ./lnets/tasks/vae/mains/train_VAE.py ./lnets/tasks/vae/configs/mnist/2D_latents_fc_VAE_bjorck.json -o model.linear.bjorck_iter=3 model.encoder_mean.l_constant=$l_constant model.encoder_std_dev.l_constant=$l_constant model.decoder.l_constant=$l_constant --saving_tag=$saving_tag
	echo "Finetuning model for Lipschitz constant $l_constant..."
	if [ "$saving_tag" == "" ]; then
		pythonw ./lnets/tasks/vae/mains/ortho_finetune.py --model.exp_path=./out/vae/mnist/not_finetuned/lipschitz_${l_constant}_latent_dim_2
	else 
		pythonw ./lnets/tasks/vae/mains/ortho_finetune.py --model.exp_path=./out/vae/mnist/not_finetuned/lipschitz_${l_constant}_latent_dim_2_${saving_tag} --saving_tag=${saving_tag}
	fi
done