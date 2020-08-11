# BB: Written starting July 11 and based heavily on classification_model.py

import torch
import torch.nn.functional as functional
from torch.autograd import Variable
import torchnet as tnt

from lnets.models.model_types.base_model import ExperimentModel

class VAEMNISTModel(ExperimentModel):
    def _init_meters(self):
        super(VAEMNISTModel, self)._init_meters()

    def loss(self, sample, test=False, check_likelihood=False):
        inputs = Variable(sample[0], volatile=test)
        reconstructions, encoder_mean, encoder_variance = self.model.forward(inputs)
        reshaped_inputs = inputs.reshape(reconstructions.shape)
        KL_term = 0.5 * (1.0 + encoder_variance.log() - encoder_mean.pow(2) - encoder_variance).sum()
        NLLs = (-1.0) * (reshaped_inputs * reconstructions.log() + (1.0 - reshaped_inputs) * (1.0 - reconstructions).log()).sum(dim=1)
        LL_term = (reshaped_inputs * reconstructions.log() + (1.0 - reshaped_inputs) * (1.0 - reconstructions).log()).sum()
        if check_likelihood:
            return (-1.0) * (self.model.training_set_size / reshaped_inputs.shape[0]) * (KL_term + LL_term), reconstructions, NLLs
        else:
            return (-1.0) * (self.model.training_set_size / reshaped_inputs.shape[0]) * (KL_term + LL_term), reconstructions

    def add_to_meters(self, state):
        self.meters['loss'].add(state['loss'].item())