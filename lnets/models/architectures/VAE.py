# BB: VAE with a fully-connected encoder / decoder and diagonal Gaussian posterior (modeled heavily on fully_connected.py)

import torch
import torch.nn as nn
import torch.distributions as ds

from lnets.models.layers import *
from lnets.models.utils import *
from lnets.models.architectures.base_architecture import Architecture

class fcMNISTVAE(Architecture):
    def __init__(self, encoder_mean_layers, encoder_st_dev_layers, decoder_layers, input_dim, latent_dim, linear_type, activation, bias=True, config=None, dropout=False):
        super(fcMNISTVAE, self).__init__()
        self.config = config

        # Store size of training set for loss computation purposes.
        self.training_set_size = self.config.data.training_set_size

        self.input_dim = input_dim 
        self.latent_dim = latent_dim

        self.encoder_mean_layer_sizes = encoder_mean_layers.copy()
        self.encoder_mean_layer_sizes.insert(0, self.input_dim)  # For bookkeeping purposes.
        self.encoder_st_dev_layer_sizes = encoder_st_dev_layers.copy()
        self.encoder_st_dev_layer_sizes.insert(0, self.input_dim)  # For bookkeeping purposes.
        self.decoder_layer_sizes = decoder_layers.copy()
        self.decoder_layer_sizes.insert(0, self.latent_dim) # For bookkeeping purposes.

        self.encoder_mean_l_constant = self.config.model.encoder_mean.l_constant
        self.encoder_st_dev_l_constant = self.config.model.encoder_st_dev.l_constant
        self.decoder_l_constant = self.config.model.decoder.l_constant
        
        self.encoder_mean_num_layers = len(self.encoder_mean_layer_sizes)
        self.encoder_st_dev_num_layers = len(self.encoder_st_dev_layer_sizes)
        self.decoder_num_layers = len(self.decoder_layer_sizes)

        # Select activation function and grouping.
        self.act_func = select_activation_function(activation)

        if 'groupings' in self.config.model.encoder_mean:
            self.encoder_mean_groupings = self.config.model.encoder_mean.groupings
            self.encoder_mean_groupings.insert(0, -1)  # For easier bookkeeping later on.

        if 'groupings' in self.config.model.encoder_st_dev:
            self.encoder_st_dev_groupings = self.config.model.encoder_st_dev.groupings
            self.encoder_st_dev_groupings.insert(0, -1)  # For easier bookkeeping later on.  

        if 'groupings' in self.config.model.decoder:
            self.decoder_groupings = self.config.model.decoder.groupings
            self.decoder_groupings.insert(0, -1)  # For easier bookkeeping later on.  

        # Select linear layer type.
        self.linear_type = linear_type
        self.use_bias = bias
        self.linear = select_linear_layer(self.linear_type)

        encoder_mean_layers = self._get_sequential_layers(activation=activation,
                                             l_constant_per_layer=self.encoder_mean_l_constant ** (1.0 / (self.encoder_mean_num_layers - 1)),
                                             config=config, dropout=dropout, function='encoder_mean')
        self.encoder_mean = nn.Sequential(*encoder_mean_layers)

        encoder_st_dev_layers = self._get_sequential_layers(activation=activation,
                                             l_constant_per_layer=self.encoder_st_dev_l_constant ** (1.0 / (self.encoder_st_dev_num_layers - 1)),
                                             config=config, dropout=dropout, function='encoder_st_dev')
        self.encoder_st_dev = nn.Sequential(*encoder_st_dev_layers)
        
        decoder_layers = self._get_sequential_layers(activation=activation,
                                             l_constant_per_layer=self.decoder_l_constant ** (1.0 / (self.decoder_num_layers - 1)),
                                             config=config, dropout=dropout, function='decoder')
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.standard_normal = ds.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        encoder_mean = self.encoder_mean(x)
        encoder_st_dev = self.encoder_st_dev(x)
        z = encoder_mean + encoder_st_dev * self.standard_normal.sample(encoder_mean.shape)
        return self.decoder(z), encoder_mean, encoder_st_dev


    def _get_sequential_layers(self, activation, l_constant_per_layer, config, dropout=False, function=None):
        # First linear transformation.
        # Add layerwise output scaling to control the Lipschitz Constant of the whole network.
        layers = list()
        if dropout:
            layers.append(nn.Dropout(0.2))
        layers.append(self.linear(eval('self.' + function + '_layer_sizes')[0], eval('self.' + function + '_layer_sizes')[1], bias=self.use_bias, config=config))
        layers.append(Scale(l_constant_per_layer, cuda=self.config.cuda))

        for i in range(1, len(eval('self.' + function + '_layer_sizes')) - 1):
            # Determine the downsampling that happens after each activation.
            if activation == "maxout":
                downsampling_factor = (1.0 / eval('self.' + function + '_groupings')[i])
            elif activation == "maxmin" or activation == "norm_twist":
                downsampling_factor = (2.0 / eval('self.' + function + '_groupings')[i])
            else:
                downsampling_factor = 1.0

            # Add the activation function.
            if activation in ["maxout", "maxmin", "group_sort", "norm_twist"]:
                layers.append(self.act_func(eval('self.' + function + '_layer_sizes')[i] // eval('self.' + function + '_groupings')[i]))
            else:
                layers.append(self.act_func())

            if dropout:
                layers.append(nn.Dropout(0.5))

            # Add the linear transformations.
            layers.append(
                self.linear(int(downsampling_factor * eval('self.' + function + '_layer_sizes')[i]), eval('self.' + function + '_layer_sizes')[i + 1], bias=self.use_bias,
                            config=config))
            layers.append(Scale(l_constant_per_layer, cuda=self.config.cuda))

            # BB: Need to return and adjust scaling to ensure control over Lipschitz constant of encoder st. dev. network
            if function != 'encoder_mean' and i == (len(eval('self.' + function + '_layer_sizes')) - 2):
                layers.append(nn.Sigmoid())

        return layers

    def project_network_weights(self, proj_config):
        # Project the weights on the manifold of orthonormal matrices.
        for i, layer in enumerate(self.encoder_mean):
            if hasattr(self.encoder_mean[i], 'project_weights'):
                self.encoder_mean[i].project_weights(proj_config)

        for i, layer in enumerate(self.encoder_st_dev):
            if hasattr(self.encoder_st_dev[i], 'project_weights'):
                self.encoder_st_dev[i].project_weights(proj_config)

        for i, layer in enumerate(self.decoder):
            if hasattr(self.decoder[i], 'project_weights'):
                self.decoder[i].project_weights(proj_config)

    # BB: Not implemented for now (left until later / necessary)
    def get_activations(self, x):
        raise NotImplementedError