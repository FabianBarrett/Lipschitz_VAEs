import torch.nn as nn

from lnets.models.layers import DenseLinear, StandardLinear
from lnets.models.layers import BjorckLinear
from lnets.utils.math.projections import bjorck_orthonormalize


def convert_to_bjorck_module(module_list, config):
    for i in range(len(module_list)):
        m = module_list[i]
        if isinstance(m, DenseLinear):
            new_linear = BjorckLinear(m.in_features, m.out_features, m.bias is not None, config)
            # BB: Commented out since I think orthonormalization should be distinct from exchanging the type of layers
            # new_linear.weight.data.copy_(bjorck_orthonormalize(m.weight, iters=30))
            new_linear.weight.data.copy_(m.weight)
            new_linear.bias.data.copy_(m.bias)
            module_list[i] = new_linear
        if isinstance(m, nn.Sequential):
            module_list[i] = nn.Sequential(*convert_to_bjorck_module(list(m.children()), config))
    return module_list


def convert_from_bjorck_module(module_list, config):
    for i in range(len(module_list)):
        m = module_list[i]
        if isinstance(m, BjorckLinear):
            new_linear = StandardLinear(m.in_features, m.out_features, m.bias is not None, config)
            new_linear.weight = m.weight
            new_linear.bias = m.bias
            module_list[i] = new_linear
        if isinstance(m, nn.Sequential):
            module_list[i] = nn.Sequential(*convert_from_bjorck_module(list(m.children()), config))
    return module_list


def convert_model_to_bjorck(model, config):
    if not isinstance(model.model.model, nn.Sequential):
        raise Exception('Model type different. ')

    module_list = convert_to_bjorck_module(list(model.model.model.children()), config)

    model.model.model = nn.Sequential(*module_list)
    return model

def convert_model_from_bjorck(model, config):
    if not isinstance(model.model.model, nn.Sequential):
        raise Exception('Model type different. ')

    module_list = convert_from_bjorck_module(list(model.model.model.children()), config)

    model.model.model = nn.Sequential(*module_list)
    return model

def convert_VAE_to_bjorck(model, config):
    encoder_mean_list = convert_to_bjorck_module(list(model.model.encoder_mean.children()), config)
    model.model.encoder_mean = nn.Sequential(*encoder_mean_list)
    encoder_std_dev_list = convert_to_bjorck_module(list(model.model.encoder_std_dev.children()), config)
    model.model.encoder_std_dev = nn.Sequential(*encoder_std_dev_list)
    decoder_list = convert_to_bjorck_module(list(model.model.decoder.children()), config)
    model.model.decoder = nn.Sequential(*decoder_list)

    return model

def convert_VAE_from_bjorck(model, config):

    encoder_mean_list = convert_from_bjorck_module(list(model.model.encoder_mean.children()), config)
    model.model.encoder_mean = nn.Sequential(*encoder_mean_list)
    encoder_std_dev_list = convert_from_bjorck_module(list(model.model.encoder_std_dev.children()), config)
    model.model.encoder_std_dev = nn.Sequential(*encoder_std_dev_list)
    decoder_list = convert_from_bjorck_module(list(model.model.decoder.children()), config)
    model.model.decoder = nn.Sequential(*decoder_list)

    return model
