l_constants=( 10.0 )
saving_tag="continuous_bernoulli"
for l_constant in "${l_constants[@]}"
do
	echo "Training model for Lipschitz constant $l_constant..."
	pythonw ./lnets/tasks/vae/mains/train_VAE.py ./lnets/tasks/vae/configs/mnist/fc_VAE_bjorck.json -o model.linear.bjorck_iter=3 model.encoder_mean.l_constant=$l_constant model.encoder_variance.l_constant=$l_constant model.decoder.l_constant=$l_constant --saving_tag=$saving_tag
done