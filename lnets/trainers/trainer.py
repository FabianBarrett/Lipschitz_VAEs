import torch
# from lnets.models.layers import BjorckLinear

"""
Based on code from https://github.com/pytorch/tnt/blob/master/torchnet/trainers/trainers.py
"""

class Trainer(object):
    def __init__(self):
        self.hooks = {}

    def hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def train(self, model, iterator, maxepoch, optimizer, convergence_tol=None):
        # Initialize the state that will fully describe the status of training.
        state = {
            'model': model,
            'iterator': iterator,
            'maxepoch': maxepoch,
            'optimizer': optimizer,
            'epoch': 0,
            't': 0,
            'train': True,
            'stop': False,
            'previous_epoch_loss': None, 
            'convergence_tol': convergence_tol
        }

        # On training start.
        model.train()  # Switch to training mode.
        self.hook('on_start', state)

        # Loop over epochs.
        while state['epoch'] < state['maxepoch'] and not state['stop']:
            # On epoch start.
            self.hook('on_start_epoch', state)

            # Loop over samples.
            for sample in state['iterator']:

                # On sample.
                state['sample'] = sample
                self.hook('on_sample', state)

                def closure():
                    loss, output = state['model'].loss(state['sample'])
                    state['output'] = output
                    state['loss'] = loss

                    # print(50 * "*")
                    # print("Iteration: {}".format(state['t']))
                    # print("Input: {}".format(state['sample']))
                    # print("Output: {}".format(output))
                    # print("Loss: {}".format(loss))

                    if torch.isnan(loss):
                        raise RuntimeError("Loss diverged.")
                    loss.backward()

                    self.hook('on_forward', state)

                    state['previous_loss'] = loss

                    # To free memory in save_for_backward,
                    # state['output'] = None
                    # state['loss'] = None
                    return loss

                # for module in state['model'].modules():
                #     if isinstance(module, BjorckLinear):
                #         print(30 * "+")
                #         print("Module dimensions: {}".format(module.weight.t().shape))
                #         print("Module weights: {}".format(module.weight))
                #         print("Module gradients: {}".format(module.weight.grad))
                # print(50 * "*")

                # On update.
                state['optimizer'].zero_grad()
                state['optimizer'].step(closure)
                self.hook('on_update', state)

                state['t'] += 1
            state['epoch'] += 1

            # On epoch end.
            self.hook('on_end_epoch', state)

        # On training end.
        self.hook('on_end', state)

        return state

    def test(self, model, iterator):
        # Initialize the state that will fully describe the status of training.
        state = {
            'model': model,
            'iterator': iterator,
            't': 0,
            'train': False,
        }
        model.eval()  # Set the PyTorch model to evaluation mode.

        # On start.
        self.hook('on_start', state)
        self.hook('on_start_val', state)

        # Loop over samples - for one epoch.
        for sample in state['iterator']:
            # On sample.
            state['sample'] = sample

            self.hook('on_sample', state)

            def closure():
                loss, output = state['model'].loss(state['sample'], test=True)
                state['output'] = output
                state['loss'] = loss
                self.hook('on_forward', state)
                # To free memory in save_for_backward.
                # state['output'] = None
                # state['loss'] = None

            closure()
            state['t'] += 1

        # On training end.
        self.hook('on_end_val', state)
        self.hook('on_end', state)
        model.train()
        return state
