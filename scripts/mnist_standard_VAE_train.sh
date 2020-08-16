# Make sure to include standard in saving_tag to ensure proper naming convention for standard VAE saving
saving_tag="standard_continuous_bernoulli"
pythonw ./lnets/tasks/vae/mains/train_VAE.py ./lnets/tasks/vae/configs/mnist/fc_VAE_standard.json --saving_tag=${saving_tag}