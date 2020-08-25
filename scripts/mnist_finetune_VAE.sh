# This code has been modified for the purpose of experimentation; do not use out-of-the-box / comment out certain args
l_constants=( 5.0 )
saving_tag="continuous_bernoulli_std_dev_better_convergence"
for l_constant in "${l_constants[@]}"
do
	echo "Finetuning model for Lipschitz constant $l_constant..."
	if [ "$saving_tag" == "" ]; then
		pythonw ./lnets/tasks/vae/mains/ortho_finetune.py --model.exp_path=./out/vae/mnist/not_finetuned/lipschitz_${l_constant}
	else 
		pythonw ./lnets/tasks/vae/mains/ortho_finetune.py --model.exp_path=./out/vae/mnist/not_finetuned/lipschitz_${l_constant}_latent_dim_2_${saving_tag} --saving_tag=${saving_tag}
	fi
done