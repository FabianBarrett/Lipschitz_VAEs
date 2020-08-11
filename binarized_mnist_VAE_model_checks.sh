l_constants=( 2.0 )
for l_constant in "${l_constants[@]}"
do
	echo "Checking properties of VAE with Lipschitz constant $l_constant..."
	pythonw ./lnets/tasks/vae/mains/model_checks.py --model.exp_path=./out/vae/binarized_mnist/not_finetuned/lipschitz_$l_constant --ortho_iters=50
done