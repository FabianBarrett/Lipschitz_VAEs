# This code has been modified for the purpose of experimentation; do not use out-of-the-box / comment out certain args
l_constants=( 15.0 )
saving_tag="continuous_bernoulli_st_dev"
for l_constant in "${l_constants[@]}"
do
	echo "Finetuning model for Lipschitz constant $l_constant..."
	if [ "$saving_tag" == "" ]; then
		pythonw ./lnets/tasks/vae/mains/ortho_finetune.py --model.exp_path=./out/vae/mnist/not_finetuned/lipschitz_${l_constant} --max_grad_norm=1e2 --ortho_iters=3
	else 
		pythonw ./lnets/tasks/vae/mains/ortho_finetune.py --model.exp_path=./out/vae/mnist/not_finetuned/lipschitz_${l_constant}_${saving_tag} --saving_tag=${saving_tag} --max_grad_norm=1e2 --ortho_iters=3
	fi
done