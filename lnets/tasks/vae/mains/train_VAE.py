# BB: Written starting July 11 
# BB: Based on train_classifier.py
# Training function for VAEs

from functools import partial
from tqdm import tqdm
import sys
import numpy as np

from lnets.utils.config import process_config
from lnets.data.load_data import load_data
from lnets.trainers.trainer import Trainer
from lnets.utils.logging import Logger
from lnets.utils.training_getters import get_optimizer, get_scheduler
from lnets.utils.saving_and_loading import *
from lnets.utils.seeding import set_experiment_seed
from lnets.utils.misc import *
from lnets.utils.training_getters import get_training_dirs
from lnets.tasks.vae.mains.utils import visualize_reconstructions

def train(model, loaders, config, finetune=False, saving_tag=""):
    # Set the seed.
    set_experiment_seed(config.seed)

    # Get relevant paths.
    dirs = get_training_dirs(config, VAE=True, finetune=finetune, saving_tag=saving_tag)

    # Get optimizer and learning rate scheduler.
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_scheduler(config, optimizer)

    # Load pretrained model and the state of the optimizer when it was saved.
    if config.model.pretrained_best_path:
        load_best_model_and_optimizer(model, optimizer, config.model.pretrained_best_path)

    # Push model to GPU if available.
    if config.cuda:
        print('Using cuda: {}'.format("Yes"))
        model.cuda()

    # Get logger, and log the config.
    logger = Logger(dirs.log_dir)
    logger.log_config(config)

    # Instantiate the trainer.
    trainer = Trainer()

    best_val = initialize_best_val(config.optim.criterion.minmax)

    # Define hooks.
    def on_sample(state):
        if config.cuda:
            state['sample'] = [x.cuda() for x in state['sample']]

    def on_forward(state):
        state['model'].add_to_meters(state)

        # Clip gradients.
        torch.nn.utils.clip_grad_norm_(state['model'].parameters(), config.optim.max_grad_norm)

    def on_update(state):
        if config.model.per_update_proj.turned_on:
            state['model'].model.project_network_weights(config.model.per_update_proj)

    def on_start(state):
        state['loader'] = state['iterator']
        state['scheduler'] = scheduler

    def on_start_epoch(state):
        state['model'].reset_meters()
        state['iterator'] = tqdm(state['loader'], desc='Epoch {}'.format(state['epoch']))

        # Project the weights on the orthonormal matrix manifold if the layer type is suitable to do so.
        if config.model.per_epoch_proj.turned_on:
            if state['epoch'] % config.model.per_epoch_proj.every_n_epochs == 0 and state['epoch'] != 0:
                state['model'].model.project_network_weights(config.model.per_epoch_proj)
                # Reset optimizer is necessary. Especially useful for stateful optimizers.
                if config.model.per_epoch_proj.reset_optimizer:
                    state['optimizer'] = get_optimizer(config, model.parameters())

    def on_end_epoch(hook_state, state):
        scheduler.step()

        print("Training loss: {:.4f}".format(state['model'].meters['loss'].value()[0]))

        state['current_epoch_loss'] = state['model'].meters['loss'].value()[0]

        if state.get('convergence_tol', None) is not None and state.get('previous_epoch_loss', None) is not None:
            if np.abs(state['current_epoch_loss'] - state['previous_epoch_loss']) <= state['convergence_tol']:
                state['stop'] = True
                print("Convergence criterion reached.")

        state['previous_epoch_loss'] = state['current_epoch_loss']

        logger.log_meters('train', state)

        if state['epoch'] % config.logging.report_freq == 0:
            if config.logging.save_model:
                save_current_model_and_optimizer(model, optimizer, model_dir=dirs.model_dir, epoch=state['epoch'])

        if config.logging.visualize:
            visualize_reconstructions(model, loaders['test'], config, figures_dir=dirs.figures_dir, epoch=state['epoch'])

        # Do validation at the end of each epoch.
        if config.data.validation:
            state['model'].reset_meters()
            trainer.test(model, loaders['validation'])
            print("Val loss: {:.4f}".format(state['model'].meters['loss'].value()[0]))
            logger.log_meters('val', state)

            # Check if this is the best model.
            if config.logging.save_best:
                hook_state['best_val'], new_best = save_best_model_and_optimizer(state, hook_state['best_val'],
                                                                                 dirs.best_path, config)

    trainer.hooks['on_start'] = on_start
    trainer.hooks['on_sample'] = on_sample
    trainer.hooks['on_forward'] = on_forward
    trainer.hooks['on_update'] = on_update
    trainer.hooks['on_start_epoch'] = on_start_epoch
    trainer.hooks['on_end_epoch'] = partial(on_end_epoch, {'best_val': best_val, 'wait': 0})

    # BB: Hacky solution to get trainer to run to convergence if config file says to do so
    if config.optim.to_convergence:
        trainer.train(model, loaders['train'], maxepoch=sys.maxsize, optimizer=optimizer, convergence_tol=config.optim.convergence_tol)
    else:
        # Enter the training loop. 
        trainer.train(model, loaders['train'], maxepoch=config.optim.epochs, optimizer=optimizer)

    # Pick the best model according to validation score and test it.
    model.reset_meters()
    best_model_path = os.path.join(dirs.best_path, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    return model

if __name__ == '__main__':
    # Get the config, initialize the model and construct the data loader.
    cfg, saving_tag = process_config()
    print("cfg output root", cfg.output_root)
    data_loaders = load_data(cfg)
    model_initialization = get_model(cfg)
    print(model_initialization)

    # Train.
    trained_model = train(model_initialization, data_loaders, cfg, finetune=False, saving_tag=saving_tag)