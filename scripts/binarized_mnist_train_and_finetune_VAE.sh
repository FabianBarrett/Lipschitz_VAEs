l_constants=( 1.0 )
saving_tag=""
for l_constant in "${l_constants[@]}"
do
	echo "Training model for Lipschitz constant $l_constant..."
	pythonw ./lnets/tasks/vae/mains/train_VAE.py ./lnets/tasks/vae/configs/binarized_mnist/fc_VAE_bjorck.json -o model.linear.bjorck_iter=3 model.encoder_mean.l_constant=$l_constant model.encoder_std_dev.l_constant=$l_constant model.decoder.l_constant=$l_constant --saving_tag=$saving_tag
	echo "Finetuning model for Lipschitz constant $l_constant..."
	if [ "$saving_tag" == "" ]; then
		pythonw ./lnets/tasks/vae/mains/ortho_finetune.py --model.exp_path=./out/vae/binarized_mnist/not_finetuned/lipschitz_${l_constant}
	else 
		pythonw ./lnets/tasks/vae/mains/ortho_finetune.py --model.exp_path=./out/vae/binarized_mnist/not_finetuned/lipschitz_${l_constant}_${saving_tag}
	fi
done