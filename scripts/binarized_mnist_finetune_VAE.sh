l_constants=( 10.0 15.0 20.0 25.0 30.0 35.0 )
for l_constant in "${l_constants[@]}"
do
	echo "Finetuning model for Lipschitz constant $l_constant ..."
	pythonw ./lnets/tasks/vae/mains/ortho_finetune.py --model.exp_path=./out/vae/binarized_mnist/not_finetuned/lipschitz_$l_constant --ortho_iters=4
done