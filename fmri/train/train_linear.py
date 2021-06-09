from fmri.models.supervised.cnn3d import ConvResnet3D
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from fmri.utils.activations import Swish, Mish
from fmri.utils.CycleAnnealScheduler import CycleScheduler
from fmri.utils.dataset import load_checkpoint, save_checkpoint, MRIDatasetClassifier
from fmri.utils.transform_3d import Normalize, RandomRotation3D, ColorJitter3D, Flip90, Flip180, Flip270, XFlip, YFlip, \
    ZFlip
from fmri.models.supervised.MLP import MLP, LinearClassifier
from fmri.utils.plot_performance import plot_performance
import torchvision
from torchvision import transforms
from ax.service.managed_loop import optimize
import random
from medicaltorch import transforms as mt_transforms

import os

import nibabel as nib
from fmri.utils.utils import validation_spliter

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = "cpu"


class Train:
    def __init__(self,
                 n_inputs,
                 save,
                 path,
                 n_classes,
                 init_func=torch.nn.init.kaiming_uniform_,
                 activation=torch.nn.GELU,
                 batch_size=8,
                 epochs=1000,
                 checkpoint_path=None,
                 epochs_per_checkpoint=1,
                 epochs_per_print=1,
                 batchnorm=False,
                 verbose=2,
                 size=32,
                 mean=0.5,
                 std=0.5,
                 plot_perform=True,
                 val_share=0.1,
                 cross_validation=5
                 ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        self.epochs_per_checkpoint = epochs_per_checkpoint
        self.epochs_per_print = epochs_per_print
        self.batchnorm = batchnorm
        self.save = save
        self.verbose = verbose
        self.path = path
        self.size = size
        self.std = std
        self.mean = mean
        self.activation = activation
        self.init_func = init_func
        self.val_share = val_share
        self.plot_perform = plot_perform
        self.cross_validation = cross_validation

    def train(self, params):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        mom_range = params['mom_range']
        n_res = params['n_res']
        niter = params['niter']
        scheduler = params['scheduler']
        optimizer_type = params['optimizer']
        momentum = params['momentum']
        learning_rate = params['learning_rate'].__format__('e')
        weight_decay = params['weight_decay'].__format__('e')

        weight_decay = float(str(weight_decay)[:1] + str(weight_decay)[-4:])
        learning_rate = float(str(learning_rate)[:1] + str(learning_rate)[-4:])
        if self.verbose > 1:
            print("Parameters: \n\t",
                  'zdim: ' + str(self.n_classes) + "\n\t",
                  'mom_range: ' + str(mom_range) + "\n\t",
                  'niter: ' + str(niter) + "\n\t",
                  'nres: ' + str(n_res) + "\n\t",
                  'learning_rate: ' + learning_rate.__format__('e') + "\n\t",
                  'momentum: ' + str(momentum) + "\n\t",
                  'weight_decay: ' + weight_decay.__format__('e') + "\n\t",
                  'optimizer_type: ' + optimizer_type + "\n\t",
                  )

        self.modelname = "classif_3dcnn_" \
                         + '_bn' + str(self.batchnorm) \
                         + '_niter' + str(niter) \
                         + '_nres' + str(n_res) \
                         + '_momrange' + str(mom_range) \
                         + '_momentum' + str(momentum) \
                         + '_' + str(optimizer_type) \
                         + "_nclasses" + str(self.n_classes) \
                         + '_initlr' + learning_rate.__format__('e') \
                         + '_wd' + weight_decay.__format__('e') \
                         + '_size' + str(self.size)
        model = LinearClassifier(n_inputs=self.n_inputs,
                                 n_classes=self.n_classes,
                                 activation=torch.nn.ReLU).to(device)
        criterion = nn.CrossEntropyLoss()
        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(params=model.parameters(),
                                          lr=learning_rate,
                                          weight_decay=weight_decay,
                                          amsgrad=True)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(params=model.parameters(),
                                        lr=learning_rate,
                                        weight_decay=weight_decay,
                                        momentum=momentum)
        elif optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(params=model.parameters(),
                                            lr=learning_rate,
                                            weight_decay=weight_decay,
                                            momentum=momentum)
        else:
            exit('error: no such optimizer type available')

        epoch = 0
        model = model.to(device)

        all_set = MRIDatasetClassifier(self.path, transform=None, size=self.size)
        spliter = validation_spliter(all_set, cv=self.cross_validation)

        print("Training Started on device:", device)
        best_losses = []
        for cv in range(self.cross_validation):
            model.random_init(self.init_func)
            best_loss = -1
            valid_set, train_set = spliter.__next__()
            valid_set.transform = False
            train_loader = DataLoader(train_set,
                                      num_workers=0,
                                      shuffle=True,
                                      batch_size=self.batch_size,
                                      pin_memory=False,
                                      drop_last=True)
            valid_loader = DataLoader(valid_set,
                                      num_workers=0,
                                      shuffle=True,
                                      batch_size=2,
                                      pin_memory=False,
                                      drop_last=True)

            # Get shared output_directory ready
            logger = SummaryWriter('logs')

            if scheduler == 'ReduceLROnPlateau':
                lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                         factor=0.1,
                                                                         cooldown=50,
                                                                         patience=50,
                                                                         verbose=True,
                                                                         min_lr=1e-15)
            elif scheduler == 'CycleScheduler':
                lr_schedule = CycleScheduler(optimizer,
                                             learning_rate,
                                             n_iter=niter * len(train_loader),
                                             momentum=[
                                                 max(0.0, momentum - mom_range),
                                                 min(1.0, momentum + mom_range),
                                             ])

            losses = {
                "train": [],
                "valid": [],
            }
            accuracies = {
                "train": [],
                "valid": [],
            }
            shapes = {
                "train": len(train_set),
                "valid": len(valid_set),
            }
            early_stop_counter = 0
            print("\n\n\nCV:", cv, "/", self.cross_validation, "\nTrain samples:", len(train_set),
                  "\nValid samples:", len(valid_set), "\n\n\n")
            train_losses = []
            train_accuracy = []
            valid_losses = []
            valid_accuracy = []
            for epoch in range(self.epochs):
                if early_stop_counter == 200:
                    if self.verbose > 0:
                        print('EARLY STOPPING.')
                    break
                best_epoch = False
                model.train()

                # pbar = tqdm(total=len(train_loader))
                for i, batch in enumerate(train_loader):
                    #    pbar.update(1)
                    model.zero_grad()
                    images, targets = batch

                    images = images.to(device)
                    targets = targets.to(device)

                    preds = model(images)

                    loss = criterion(preds, targets)

                    loss.backward()

                    accuracy = sum(
                        [1 if torch.argmax(pred) == target else 0 for (pred, target) in zip(preds, targets)]) / len(
                        targets)
                    train_accuracy += [accuracy]

                    train_losses += [loss.item()]

                    optimizer.step()
                    if scheduler == "CycleScheduler":
                        lr_schedule.step()
                    logger.add_scalar('training_loss', loss.item(), i + len(train_loader) * epoch)
                    del loss

                if epoch % self.epochs_per_print == 0:
                    losses["train"] += [np.mean(train_losses)]
                    accuracies["train"] += [np.mean(train_accuracy)]
                    if self.verbose > 1:
                        print("Epoch: {}:\t"
                              "Train Loss: {:.5f} , "
                              "Accuracy: {:.3f} , "
                              .format(epoch,
                                      losses["train"][-1],
                                      accuracies["train"][-1]
                                      ))
                    train_losses = []
                    train_accuracy = []

                model.eval()
                for i, batch in enumerate(valid_loader):
                    images, targets = batch
                    images = images.to(device)
                    targets = targets.to(device)
                    preds = model(images)

                    loss = criterion(preds, targets)
                    valid_losses += [loss.item()]
                    accuracy = sum(
                        [1 if torch.argmax(pred) == target else 0 for (pred, target) in zip(preds, targets)]) / len(
                        targets)
                    valid_accuracy += [accuracy]
                    logger.add_scalar('training loss', np.log2(loss.item()), i + len(train_loader) * epoch)
                if scheduler == "ReduceLROnPlateau":
                    if epoch > 25:
                        lr_schedule.step(losses["valid"][-1])
                mode = 'valid'
                if epoch > 10 and epoch % self.epochs_per_print == 0:
                    if (losses[mode][-1] < best_loss or best_loss == -1) \
                            and not np.isnan(losses[mode][-1]):
                        if self.verbose > 1:
                            print('BEST EPOCH!', losses[mode][-1], accuracies[mode][-1])
                        early_stop_counter = 0
                        best_loss = losses[mode][-1]
                        best_epoch = True
                    else:
                        early_stop_counter += 1

                if epoch % self.epochs_per_print == 0:
                    losses["valid"] += [np.mean(valid_losses)]
                    accuracies["valid"] += [np.mean(valid_accuracy)]
                    if self.verbose > 0:
                        print("Epoch: {}:\t"
                              "Valid Loss: {:.5f} , "
                              "Accuracy: {:.3f} "
                              .format(epoch,
                                      losses["valid"][-1],
                                      accuracies["valid"][-1],
                                      )
                              )
                    if self.verbose > 1:
                        print("Current LR:", optimizer.param_groups[0]['lr'])
                    if 'momentum' in optimizer.param_groups[0].keys():
                        print("Current Momentum:", optimizer.param_groups[0]['momentum'])
                    valid_losses = []
                    valid_accuracy = []

                if self.plot_perform:
                    plot_performance(loss_total=losses, losses_recon=None, accuracies=accuracies,
                                     kl_divs=None, shapes=shapes,
                                     results_path="../figures",
                                     filename="training_loss_trace_"
                                              + self.modelname + '.jpg')
            if self.verbose > 0:
                print('BEST LOSS :', best_loss)
            best_losses += [best_loss]
        return best_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.manual_seed(11)

    random.seed(10)

    size = 32
    bs = 56
    n_inputs = 32768
    checkpoint_path = "checkpoints"
    path = '/home/simon/loris-api-presentation/fmri/'

    n_epochs = 10000
    save = False
    training = Train(n_inputs=n_inputs,
                     path=path,
                     batch_size=bs,
                     epochs=n_epochs,
                     checkpoint_path=checkpoint_path,
                     epochs_per_checkpoint=1,
                     save=save,
                     activation=torch.nn.ReLU,
                     init_func=torch.nn.init.xavier_uniform_,
                     n_classes=2,
                     epochs_per_print=10,
                     size=size
                     )
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "mom_range", "type": "choice", "values": [0, 0]},
            {"name": "niter", "type": "choice", "values": [1000, 1000]},
            {"name": "n_res", "type": "range", "bounds": [0, 1]},
            {"name": "scheduler", "type": "choice", "values":
                ['ReduceLROnPlateau', 'ReduceLROnPlateau']},
            {"name": "optimizer", "type": "choice", "values": ['sgd', 'sgd']},
            {"name": "weight_decay", "type": "range", "bounds": [1e-14, 1e-1], "log_scale": True},
            {"name": "momentum", "type": "range", "bounds": [0.9, 1.]},
            {"name": "learning_rate", "type": "range", "bounds": [1e-4, 1e-3], "log_scale": True},
        ],
        evaluation_function=training.train,
        objective_name='loss',
        minimize=True,
        total_trials=100
    )
    from matplotlib import pyplot as plt

    fig = plt.figure()
    # render(plot_contour(model=model, param_x="learning_rate", param_y="weight_decay", metric_name='Loss'))
    # fig.savefig('test.jpg')
    print('Best Loss:', values[0]['loss'])
    print('Best Parameters:')
    print(json.dumps(best_parameters, indent=4))

    # cv_results = cross_validate(model)
    # render(interact_cross_validation(cv_results))
