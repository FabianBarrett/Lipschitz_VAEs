# Code for "Lipschitz VAEs: Certifiably Robust Variational Autoencoders"

*Provided in support of Ben Barrett's dissertation, in partial fulfilment of the degree of Master of Science in Statistical Science, University of Oxford, 2020.*

This repository builds on the [implementation](https://github.com/cemanil/lnets) of Lipschitz-constrained fully-connected neural networks provided by Anil et al., 2019. We extend this code base to VAEs, providing code for the architecture, learning objective, and training of VAEs with Lipschitz-constrained encoders and decoders. We also provide code to evaluate the tightness of bounds used in the derivation of our main results, along with a number of scripts for experiments involving adversarial attacks on VAEs. 

## Getting Started
The following assumes first-time use.

1. Create a conda environment and activate it:
```
conda create -n lnets python=3.5
conda activate lnets
```
2. Install PyTorch [here](https://pytorch.org/).
3. Install torchnet using:
```
pip install git+https://github.com/pytorch/tnt.git@master
```
4. Navigate to the root of the project, and run the following to install the necessary dependencies:
```
python setup.py install
```
5. Add the project root to the PYTHONPATH, using:
```
export PYTHONPATH="${PYTHONPATH}:`pwd`"
``` 

## Code Overview
Selected files and directories are highlighted below.
```
lnets
├── models
│   └── architectures
│       └── VAE.py                          "Specifies the VAE architecture." 
│   └── model_types
│       └── VAE_MNIST_model.py              "Computes the VAE learning objective for MNIST."
├── tasks
│       └── vae
│           └── mains
│               └── train_VAE.py            "Training code."
│               └── ortho_finetune.py       "Runs Bjorck Orthonormalization for more iterations."
│               └── latent_space_attack.py  "Implements a latent space attack."
│               └── max_damage_attack.py    "Implements maximum damage attacks and r-robustness margin estimation."
│               └── model_checks.py         "Verifies the Lipschitz continuity constraints."
│               └── visualize_latents.py    "Visualizes the learned encoder."
│               └── utils.py                "Plotting and margin bound computation."
│           └── configs
│               └── mnist                   "Houses configuration files for different VAEs for MNIST."
├── other_experiments
│   └── evaluate_bounds.py                  "Evaluates the tightness of intermediate steps in the derivation of Bounds 1 and 2."
├── scripts                                 "Houses miscellaneous bash scripts to train different VAEs and run experiments."
```

## Key Variables in Configuration Files
The directory ```lnets/tasks/vae/configs``` contains a number of files to configure different VAEs. Key variables in these configuration files are:
* `model.linear.type`: Options include "standard" (a standard linear layer) and "bjorck" (a linear layer involving Bjorck Orthonormalization).
* `model.latent_dim`: An integer specifying the dimension of the VAE latent space. 
* `model.*.l_constant` (where `*` is a stand-in for `encoder_mean`, `encoder_std_dev` or `decoder`): The Lipschitz constant of the relevant model component.
* `model.activation`: The default activations functions of each model component. Options include "relu" (the ReLU activation) and "groupsort" (the GroupSort activation from Anil et al., 2019).
* `model.*.layers` (where `*` is again a stand-in for `encoder_mean`, `encoder_std_dev` or `decoder`): A list specifying the hidden dimensions of the linear layers constituting the relevant model component.
