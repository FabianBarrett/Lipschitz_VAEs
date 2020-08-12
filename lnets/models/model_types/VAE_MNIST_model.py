# BB: Written starting July 11 and based heavily on classification_model.py

import torch
import torch.nn.functional as functional
from torch.autograd import Variable
import torchnet as tnt
import torch.distributions as ds

from lnets.models.model_types.base_model import ExperimentModel

class VAEMNISTModel(ExperimentModel):
    def _init_meters(self):
        super(VAEMNISTModel, self)._init_meters()

    def loss(self, sample, test=False, check_likelihood=False, continuous_bernoulli=True):
        inputs = Variable(sample[0], volatile=test)
        reconstructions, encoder_mean, encoder_variance = self.model.forward(inputs)
        reshaped_inputs = inputs.reshape(reconstructions.shape)
        KL_term = 0.5 * (1.0 + encoder_variance.log() - encoder_mean.pow(2) - encoder_variance).sum()
        if continuous_bernoulli:
            continuous_bernoulli_likelihood = ds.continuous_bernoulli.ContinuousBernoulli(probs=reconstructions)
            LL_term = continuous_bernoulli_likelihood.log_prob(reshaped_inputs).sum()
            NLLs = (-1.0) * continuous_bernoulli_likelihood.log_prob(reshaped_inputs).sum(dim=1)
        else:
            # LL_term = (reshaped_inputs * reconstructions.log() + (1.0 - reshaped_inputs) * (1.0 - reconstructions).log()).sum()
            # NLLs = (-1.0) * (reshaped_inputs * reconstructions.log() + (1.0 - reshaped_inputs) * (1.0 - reconstructions).log()).sum(dim=1)
            bernoulli_likelihood = ds.bernoulli.Bernoulli(probs=reconstructions)
            LL_term = bernoulli_likelihood.log_prob(reshaped_inputs).sum()
            NLLs = (-1.0) * bernoulli_likelihood.log_prob(reshaped_inputs).sum(dim=1)
    
        if check_likelihood:
            # return (-1.0) * (self.model.training_set_size / reshaped_inputs.shape[0]) * (KL_term + LL_term), reconstructions, NLLs
            return (-1.0 / reshaped_inputs.shape[0]) * (KL_term + LL_term), reconstructions, NLLs
        else:
            # return (-1.0) * (self.model.training_set_size / reshaped_inputs.shape[0]) * (KL_term + LL_term), reconstructions
            return (-1.0 / reshaped_inputs.shape[0]) * (KL_term + LL_term), reconstructions

    def add_to_meters(self, state):
        self.meters['loss'].add(state['loss'].item())