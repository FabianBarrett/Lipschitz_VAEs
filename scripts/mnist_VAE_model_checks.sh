l_constants=( 5.0 )
saving_tag="continuous_bernoulli_std_dev"
for l_constant in "${l_constants[@]}"
do
	echo "Checking properties of VAE with Lipschitz constant $l_constant..."
	pythonw ./lnets/tasks/vae/mains/model_checks.py --model.exp_path=./out/vae/mnist/not_finetuned/lipschitz_${l_constant}_${saving_tag} --ortho_iters=50
done