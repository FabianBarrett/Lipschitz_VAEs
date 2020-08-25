# BB: Written starting July 26
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
from lnets.models.layers import BjorckLinear, DenseLinear, StandardLinear
from lnets.utils.math.projections import bjorck_orthonormalize
from sympy.abc import x
from sympy import Poly, Interval, FiniteSet
from sympy.solvers.inequalities import solve_poly_inequality

# Note: Assumes inequality is in form expression <= 0
def solve_bound_inequality(a, b, c, r, std_dev_norm):
    expression = (c ** 2 + b ** 2) * x ** 2 + 4 * c * std_dev_norm * x + (4 * std_dev_norm ** 2 - 0.5 * (r ** 2 / a ** 2))
    return solve_poly_inequality(Poly(expression), '<=')

# Accepts sympy Interval object as input
def process_bound_inequality_result(bound_inequality_result):
    # If there is no value satisfying the bound (sympy returns empty list)
    if not bound_inequality_result:
        return 0.0
    elif isinstance(bound_inequality_result[0], Interval):
        return bound_inequality_result[0].end
    elif isinstance(bound_inequality_result[0], FiniteSet):
        values = [element for element in bound_inequality_result[0]]
        return max(values)
    else:
        raise RuntimeError("Inequality result type not recognized.")

# Adapted from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
# BB: Note that this doesn't exactly sample uniformly from the interior of a ball with the specified radius (due to scaling at the end)
def sample_d_ball(d, radius):
    u = np.random.randn(d)
    norm = (u ** 2).sum() ** 0.5
    r = np.random.random() ** (1.0 / d)
    sample = (r * u) / norm
    scaling = np.random.uniform(1e-8, radius)
    return scaling * sample

def get_target_image(batch, input_class, input_index, num_images):
    image_counter = 0
    image_index = input_index + 1
    while image_counter < (num_images - 1):
        image_index %= num_images
        if batch[1][image_index] != input_class:
            return batch[0][image_index], batch[1][image_index]
        image_counter += 1
        image_index += 1
    raise RuntimeError("No appropriate target image found.")

# Weird required hack to fix groupings
def fix_groupings(config):

    if 'groupings' in config.model.encoder_mean and config.model.encoder_mean.groupings[0] is -1:
        config.model.encoder_mean.groupings = config.model.encoder_mean.groupings[1:]

    if 'groupings' in config.model.encoder_std_dev and config.model.encoder_std_dev.groupings[0] is -1:
        config.model.encoder_std_dev.groupings = config.model.encoder_std_dev.groupings[1:]

    if 'groupings' in config.model.decoder and config.model.decoder.groupings[0] is -1:
        config.model.decoder.groupings = config.model.decoder.groupings[1:]

    return config

def check_NLL(model, iterator):
    NLLs = torch.empty(0)
    for batch in iterator:
        NLLs = torch.cat((NLLs, model.loss(batch, check_likelihood=True)[2]))
    print("Mean NLL on test set: {}".format(NLLs.mean()))

def visualize_reconstructions(model, iterator, config, figures_dir=None, epoch=None, title_string=None):
    
    sample = next(iter(iterator))
    _, output = model.loss(sample)

    # Compare the original images and their reconstructions from a test batch
    original_images = sample[0].squeeze()
    reconstructions = output.view(sample[0].shape[0], config.data.im_height, -1)

    fig, ax = plt.subplots(config.logging.num_images, 2, figsize=(6, config.logging.num_images))
    for i in range(config.logging.num_images):
        ax[i, 0].imshow(original_images[i, :].detach())
        ax[i, 0].axis('off')
        ax[i, 1].imshow(reconstructions[i, :].detach())
        ax[i, 1].axis('off')
    if title_string is not None:
        fig.suptitle("{} \n Original (left) and reconstructed (right) images".format(title_string))
    else:
        fig.suptitle("Original (left) and reconstructed (right) images")
    if figures_dir is not None:
        plt.savefig(os.path.join(figures_dir, "reconstructions_epoch_{}.png".format(epoch)))
    else:
        plt.show()

def orthonormalize_layers(module_list, model_config, iters):
    for i in range(len(module_list)):
        m = module_list[i]
        if isinstance(m, DenseLinear):
            new_linear = StandardLinear(m.in_features, m.out_features, m.bias is not None, model_config)
            new_linear.weight.data = bjorck_orthonormalize(m.weight, iters=iters)
            new_linear.bias = m.bias
            module_list[i] = new_linear
        if isinstance(m, nn.Sequential):
            module_list[i] = nn.Sequential(*orthonormalize_layers(list(m.children()), model_config, iters))
    return module_list

def orthonormalize_model(model, model_config, iters=20):

    encoder_mean_list = orthonormalize_layers(list(model.model.encoder_mean.children()), model_config, iters)
    model.model.encoder_mean = nn.Sequential(*encoder_mean_list)
    encoder_std_dev_list = orthonormalize_layers(list(model.model.encoder_std_dev.children()), model_config, iters)
    model.model.encoder_std_dev = nn.Sequential(*encoder_std_dev_list)
    decoder_list = orthonormalize_layers(list(model.model.decoder.children()), model_config, iters)
    model.model.decoder = nn.Sequential(*decoder_list)

    return model

def check_module_singular_values(module_list):

    layer_counter = 1
    for i in range(len(module_list)):
        if isinstance(module_list[i], StandardLinear):
            layer_weights = module_list[i].weight.t()
            layer_weights_singular_values = np.linalg.svd(layer_weights.detach().numpy(), compute_uv=False)
            max_singular_value, min_singular_value, mean_singular_value = max(layer_weights_singular_values), min(layer_weights_singular_values), np.mean(layer_weights_singular_values)
            print("Singular values of layer {}: Max = {:.3f} | min = {:.3f} | mean = {:.3f}".format(layer_counter, max_singular_value, min_singular_value, mean_singular_value))
            layer_counter += 1 

# BB: Should only be applied to model that has already been converted from Bjorck to standard linear layers
def check_VAE_singular_values(model):

    encoder_mean_list = list(model.model.encoder_mean.children())
    print("Checking singular values of layers in the encoder mean...")
    check_module_singular_values(encoder_mean_list)

    print("Checking singular values of layers in the encoder st. dev....")
    encoder_std_dev_list = list(model.model.encoder_std_dev.children())
    check_module_singular_values(encoder_std_dev_list)

    print("Checking singular values of layers in the decoder...")
    decoder_list = list(model.model.decoder.children())
    check_module_singular_values(decoder_list)

def check_module_orthonormality(module_list, tol, verbose):

    layer_counter = 1
    for i in range(len(module_list)):
        if isinstance(module_list[i], StandardLinear):
            print("Layer weights shape: {}".format(module_list[i].weight.shape))
            print("Transposed layer weights shape: {}".format(module_list[i].weight.t().shape))
            layer_weights = module_list[i].weight.t()
            layer_weights_product_diagonal = layer_weights.t().mm(layer_weights).diag()
            if verbose:
                print("Layer {} weights product diagonal: {}".format(layer_counter, layer_weights_product_diagonal))
            # BB: Checks whether all diagonal entries of weights product are close to 1
            layer_weights_orthonormality_boolean = layer_weights_product_diagonal.isclose(torch.ones(layer_weights_product_diagonal.shape), atol=tol).all()
            print("Layer {} weights are orthonormal? {}".format(layer_counter, layer_weights_orthonormality_boolean))
            layer_counter += 1

def check_VAE_orthonormality(model, tol=1e-2, verbose=False):

    encoder_mean_list = list(model.model.encoder_mean.children())
    print("Checking orthonormality of layers in the encoder mean at tolerance {}...".format(tol))
    check_module_orthonormality(encoder_mean_list, tol, verbose)

    print("Checking orthonormality of layers in the encoder st. dev. at tolerance {}...".format(tol))
    encoder_std_dev_list = list(model.model.encoder_std_dev.children())
    check_module_orthonormality(encoder_std_dev_list, tol, verbose)

    print("Checking orthonormality of layers in the decoder at tolerance {}...".format(tol))
    decoder_list = list(model.model.decoder.children())
    check_module_orthonormality(decoder_list, tol, verbose)
