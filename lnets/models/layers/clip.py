import torch
import torch.nn as nn


class Clip(nn.Module):
    """Constrains the input vector to have norm at most max_norm."""

    def __init__(self, max_norm, cuda=False):
        super(Clip, self).__init__()
        self.max_norm = max_norm

        if cuda:
            self.max_norm = torch.Tensor([self.max_norm]).cuda()

    def reset_parameters(self):
        pass

    def forward(self, input):
        clipped_input =  self.max_norm * (input / input.norm(p=2, dim=1)[:, None])
        clipped_output = torch.zeros(input.shape)
        # Where input rows have norm larger than max_norm, replace with clipped rows
        clipped_output[(input.norm(p=2, dim=1) > self.max_norm).nonzero().squeeze(), :] = clipped_input[(input.norm(p=2, dim=1) > self.max_norm).nonzero().squeeze(), :]
        # Where input rows have norm less than or equal to max_norm, retain original rows
        clipped_output[(input.norm(p=2, dim=1) <= self.max_norm).nonzero().squeeze(), :] = input[(input.norm(p=2, dim=1) <= self.max_norm).nonzero().squeeze(), :]
        return clipped_output

    def extra_repr(self):
        return 'max_norm={}'.format(self.max_norm)
