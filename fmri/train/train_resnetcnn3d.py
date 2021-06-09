from fmri.models.supervised.cnn3d import ConvResnet3D
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from scipy.stats import norm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from fmri.utils.activations import Swish, Mish
from fmri.utils.CycleAnnealScheduler import CycleScheduler
from fmri.utils.dataset import load_checkpoint, save_checkpoint, MRIDatasetClassifier, CTDataset
from fmri.utils.transform_3d import Normalize, RandomRotation3D, ColorJitter3D, Flip90, Flip180, Flip270, XFlip, YFlip, \
    ZFlip, RandomAffine3D
from fmri.models.supervised.resnetcnn3d import ConvResnet3D
from fmri.utils.plot_performance import plot_performance
import torchvision
from torchvision import transforms
from ax.service.managed_loop import optimize
import random
import math
import os

import nibabel as nib
from fmri.utils.utils import validation_spliter

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = "cpu"


def log_gaussian(x, mu, log_var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and log_var evaluated at x.

    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = - 0.5 * torch.log(2 * torch.tensor(math.pi, requires_grad=True)) - log_var / 2 - (x - mu) ** 2 / (
                2 * torch.exp(log_var))
    return torch.sum(log_pdf, dim=-1)


def log_gaussian(x, mu, sigma):
    return -0.5 * torch.log(2 * torch.tensor(math.pi, requires_grad=True)) - torch.log(torch.abs_(sigma)) - (x - mu) ** 2 / (2 * sigma ** 2)


class Train:
    def __init__(self,
                 train_csv,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 strides,
                 dilatations,
                 save,
                 padding,
                 path,
                 n_classes,
                 n_kernels,
                 init_func=torch.nn.init.kaiming_uniform_,
                 activation=torch.nn.GELU,
                 batch_size=8,
                 epochs=1000,
                 fp16_run=False,
                 checkpoint_path=None,
                 epochs_per_checkpoint=1,
                 epochs_per_print=1,
                 gated=True,
                 has_dense=True,
                 batchnorm=False,
                 resblocks=False,
                 maxpool=3,
                 verbose=2,
                 size=32,
                 mean=0.5,
                 std=0.5,
                 plot_perform=True,
                 val_share=0.1,
                 cross_validation=5,
                 is_bayesian=True,
                 random_node='output',
                 ):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.dilatations = dilatations
        self.padding = padding
        self.batch_size = batch_size
        self.epochs = epochs
        self.fp16_run = fp16_run
        self.checkpoint_path = checkpoint_path
        self.epochs_per_checkpoint = epochs_per_checkpoint
        self.epochs_per_print = epochs_per_print
        self.gated = gated
        self.has_dense = has_dense
        self.batchnorm = batchnorm
        self.resblocks = resblocks
        self.maxpool = maxpool
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
        self.is_bayesian = is_bayesian
        self.train_csv = train_csv
        self.n_kernels = n_kernels

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
                         + '_gated' + str(self.gated) \
                         + '_resblocks' + str(self.resblocks) \
                         + '_initlr' + learning_rate.__format__('e') \
                         + '_wd' + weight_decay.__format__('e') \
                         + '_size' + str(self.size)
        model = ConvResnet3D(self.maxpool,
                             self.in_channels,
                             self.out_channels,
                             self.kernel_sizes,
                             self.strides,
                             self.dilatations,
                             self.padding,
                             self.batchnorm,
                             self.n_classes,
                             max_fvc=None,
                             n_kernels=self.n_kernels,
                             is_bayesian=self.is_bayesian,
                             activation=torch.nn.ReLU,
                             n_res=n_res,
                             gated=self.gated,
                             has_dense=self.has_dense,
                             resblocks=self.resblocks,
                             ).to(device)
        criterion = nn.MSELoss()
        l1 = nn.L1Loss()
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
        # if self.fp16_run:
        #     from apex import amp
        #    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

        # Load checkpoint if one exists
        epoch = 0
        best_loss = -1
        if self.checkpoint_path is not None and self.save:
            model, optimizer, \
            epoch, losses, \
            kl_divs, losses_recon, \
            best_loss = load_checkpoint(self.checkpoint_path,
                                        model,
                                        self.maxpool,
                                        save=self.save,
                                        padding=self.padding,
                                        has_dense=self.has_dense,
                                        batchnorm=self.batchnorm,
                                        flow_type=None,
                                        padding_deconv=None,
                                        optimizer=optimizer,
                                        z_dim=self.n_classes,
                                        gated=self.gated,
                                        in_channels=self.in_channels,
                                        out_channels=self.out_channels,
                                        kernel_sizes=self.kernel_sizes,
                                        kernel_sizes_deconv=None,
                                        strides=self.strides,
                                        strides_deconv=None,
                                        dilatations=self.dilatations,
                                        dilatations_deconv=None,
                                        name=self.modelname,
                                        n_res=n_res,
                                        resblocks=self.resblocks,
                                        h_last=None,
                                        n_elements=None,
                                        n_flows=None,
                                        predict=False,
                                        n_kernels=self.n_kernels
                                        )
        model = model.to(device)
        # t1 = torch.Tensor(np.load('/run/media/simon/DATA&STUFF/data/biology/arrays/t1.npy'))
        # targets = torch.Tensor([0 for _ in t1])

        train_transform = transforms.Compose([
            transforms.RandomChoice([
                XFlip(),
                YFlip(),
                ZFlip()
            ]),
            transforms.RandomChoice([
                Flip90(),
                Flip180(),
                Flip270()
            ]),
            # ColorJitter3D(.1, .1, .1, .1),
            # transforms.RandomChoice(
            #    [
            #        RandomAffine3D(0, [.1, .1], [.1, .1], [.1, .1]),
            #        RandomAffine3D(1, [.1, .1], [.1, .1], [.1, .1]),
            #        RandomAffine3D(2, [.1, .1], [.1, .1], [.1, .1])
            #    ]
            # ),
            transforms.RandomChoice(
                [
                    RandomRotation3D(25, 0),
                    RandomRotation3D(25, 1),
                    RandomRotation3D(25, 2)
                ]
            ),

            torchvision.transforms.Normalize(mean=(self.mean), std=(self.std)),
            # Normalize()
        ])
        """
        """
        all_set = CTDataset(self.path, self.train_csv,
                            transform=train_transform, size=self.size)
        spliter = validation_spliter(all_set, cv=self.cross_validation)
        model.max_fvc = all_set.max_fvc
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
                                                                         cooldown=10,
                                                                         patience=20,
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
            log_gaussians = {
                "train": [],
                "valid": [],
            }
            vars = {
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
            train_log_gauss = []
            valid_log_gauss = []
            train_var = []
            valid_var = []
            valid_accuracy = []
            for epoch in range(self.epochs):
                if early_stop_counter == 100:
                    if self.verbose > 0:
                        print('EARLY STOPPING.')
                    break
                best_epoch = False
                model.train()

                # pbar = tqdm(total=len(train_loader))
                for i, batch in enumerate(train_loader):
                    #    pbar.update(1)
                    model.zero_grad()
                    _, images, targets, patient_info = batch
                    images = images.to(device)
                    targets = targets.type(torch.FloatTensor).to(device)
                    patient_info = patient_info.to(device)

                    _, mu, log_var = model(images, patient_info)
                    mu = mu.type(torch.FloatTensor).to(device)
                    log_var = log_var.type(torch.FloatTensor).to(device)

                    rv = norm(mu.detach().cpu().numpy(), np.exp(log_var.detach().cpu().numpy()))
                    train_log_gauss += [rv.pdf(mu.detach().cpu().numpy())]
                    # loss = criterion(preds, targets.to(device)) # - 0.01 * log_gaussian(preds.view(-1), mu.view(-1), log_var.view(-1))
                    loss = -log_gaussian(targets, mu, log_var) / self.batch_size
                    loss = torch.sum(loss, 0)
                    argmin = torch.argmin(loss)
                    # print('argmin: ', argmin)
                    loss = torch.mean(loss)
                    # loss += criterion(mu[argmin], log_var[argmin])
                    train_var += [np.exp(log_var[argmin].detach().cpu().numpy()) * model.max_fvc]
                    loss.backward()
                    l1_loss = l1(mu[argmin].to(device), targets.to(device))

                    accuracy = l1_loss.item()
                    train_accuracy += [accuracy * model.max_fvc]

                    train_losses += [loss.item()]

                    optimizer.step()
                    if scheduler == "CycleScheduler":
                        lr_schedule.step()
                    logger.add_scalar('training_loss', loss.item(), i + len(train_loader) * epoch)
                    del loss

                if epoch % self.epochs_per_print == 0:
                    losses["train"] += [np.mean(train_losses) / self.batch_size]
                    accuracies["train"] += [np.mean(train_accuracy)]
                    log_gaussians["train"] += [np.mean(train_log_gauss)]
                    vars['train'] += [np.mean(train_var)]
                    if self.verbose > 1:
                        print("Epoch: {}:\t"
                              "Train Loss: {:.5f} , "
                              "Accuracy: {:.3f} , "
                              "confidence: {:.9f} , "
                              "Vars: {:.9f} "
                              .format(epoch,
                                      losses["train"][-1],
                                      accuracies["train"][-1],
                                      log_gaussians["train"][-1],
                                      np.sqrt(vars["train"][-1])
                                      ))
                    train_losses = []
                    train_accuracy = []
                    train_log_gauss = []
                    train_var = []

                model.eval()
                # pbar = tqdm(total=len(valid_loader))
                for i, batch in enumerate(valid_loader):
                    #    pbar.update(1)
                    _, images, targets, patient_info = batch
                    images = images.to(device)
                    targets = targets.to(device)
                    patient_info = patient_info.to(device)
                    _, mu, log_var = model(images, patient_info)
                    rv = norm(mu.detach().cpu().numpy(), np.exp(log_var.detach().cpu().numpy()))
                    loss = -log_gaussian(targets.type(torch.FloatTensor).to(device),
                                         mu.type(torch.FloatTensor).to(device),
                                         torch.exp(log_var.type(torch.FloatTensor).to(device))) / self.batch_size
                    loss = torch.sum(loss, 0)
                    argmin = torch.argmin(loss)
                    loss = torch.mean(loss)
                    valid_losses += [np.exp(-loss.item())]
                    valid_log_gauss += [rv.pdf(mu.detach().cpu().numpy())]
                    valid_var += [np.exp(log_var[argmin].detach().cpu().numpy()) * model.max_fvc]
                    l1_loss = l1(mu[argmin], targets.to(device))

                    accuracy = l1_loss.item()
                    valid_accuracy += [accuracy * model.max_fvc]
                    logger.add_scalar('training loss', loss.item(), i + len(train_loader) * epoch)
                if scheduler == "ReduceLROnPlateau":
                    if epoch > 25:
                        lr_schedule.step(losses["valid"][-1])
                if epoch % self.epochs_per_print == 0:
                    losses["valid"] += [np.mean(valid_losses) / 2]
                    accuracies["valid"] += [np.mean(valid_accuracy)]
                    log_gaussians["valid"] += [np.mean(valid_log_gauss)]
                    vars['valid'] += [np.mean(valid_var)]
                    if self.verbose > 0:
                        print("Epoch: {}:\t"
                              "Valid Loss: {:.5f} , "
                              "Accuracy: {:.3f} "
                              "confidence: {:.9f} "
                              "Vars: {:.9f} "
                              .format(epoch,
                                      losses["valid"][-1],
                                      accuracies["valid"][-1],
                                      log_gaussians["valid"][-1],
                                      np.sqrt(vars['valid'][-1]),
                                      )
                              )
                    if self.verbose > 1:
                        print("Current LR:", optimizer.param_groups[0]['lr'])
                    if 'momentum' in optimizer.param_groups[0].keys():
                        print("Current Momentum:", optimizer.param_groups[0]['momentum'])
                    valid_losses = []
                    valid_accuracy = []
                    valid_log_gauss = []
                    valid_var = []

                mode = 'valid'
                if epoch > 1 and epoch % self.epochs_per_print == 0:
                    if (losses[mode][-1] > best_loss or best_loss == -1) \
                            and not np.isnan(losses[mode][-1]):
                        if self.verbose > 1:
                            print('BEST EPOCH!', losses[mode][-1], accuracies[mode][-1])
                        early_stop_counter = 0
                        best_loss = losses[mode][-1]
                        best_epoch = True
                    else:
                        early_stop_counter += 1

                if epoch % self.epochs_per_checkpoint == 0:
                    if best_epoch and self.save:
                        if self.verbose > 1:
                            print('Saving model...')
                        save_checkpoint(model=model,
                                        optimizer=optimizer,
                                        maxpool=self.maxpool,
                                        padding=self.padding,
                                        padding_deconv=None,
                                        learning_rate=learning_rate,
                                        epoch=epoch,
                                        checkpoint_path=checkpoint_path,
                                        z_dim=self.n_classes,
                                        gated=self.gated,
                                        batchnorm=self.batchnorm,
                                        losses=losses,
                                        kl_divs=None,
                                        losses_recon=None,
                                        in_channels=self.in_channels,
                                        out_channels=self.out_channels,
                                        kernel_sizes=self.kernel_sizes,
                                        kernel_sizes_deconv=None,
                                        strides=self.strides,
                                        strides_deconv=None,
                                        dilatations=self.dilatations,
                                        dilatations_deconv=None,
                                        best_loss=best_loss,
                                        save=self.save,
                                        name=self.modelname,
                                        n_flows=None,
                                        flow_type=None,
                                        n_res=n_res,
                                        resblocks=resblocks,
                                        h_last=None,
                                        n_elements=None,
                                        n_kernels=self.n_kernels
                                        )

                if self.plot_perform:
                    plot_performance(loss_total=losses, losses_recon=None, accuracies=accuracies,
                                     kl_divs=None, shapes=shapes,
                                     results_path="../figures",
                                     filename="training_loss_trace_"
                                              + self.modelname + '.jpg')
            if self.verbose > 0:
                print('BEST LOSS :', best_loss)
            best_losses += [best_loss]
        return np.mean(best_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.manual_seed(11)

    random.seed(10)

    size = 32
    in_channels = [1, 64, 128, 256, 512]
    out_channels = [64, 128, 256, 512, 512]
    kernel_sizes = [3, 3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1, 1]
    dilatations = [1, 1, 1, 1, 1, 1]
    paddings = [1, 1, 1, 1, 1, 1]
    bs = 8
    maxpool = 2
    has_dense = True
    batchnorm = True
    gated = False
    resblocks = True
    checkpoint_path = "/run/media/simon/DATA&STUFF/data/checkpoints"
    # if checkpoint_path not in os.listdir("/run/media/simon/DATA&STUFF/data/"):
    #    os.mkdir(checkpoint_path)
    path = '/run/media/simon/DATA&STUFF/data/train_32x32/'
    basedir = '/run/media/simon/DATA&STUFF/data/'
    path = basedir + '/train' + '_' + str(size) + 'x' + str(size) + '/'

    n_epochs = 5000
    save = True
    train_csv = '/run/media/simon/DATA&STUFF/data/train.csv'
    training = Train(train_csv,
                     in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_sizes=kernel_sizes,
                     strides=strides,
                     dilatations=dilatations,
                     path=path,
                     padding=paddings,
                     batch_size=bs,
                     epochs=n_epochs,
                     checkpoint_path=checkpoint_path,
                     epochs_per_checkpoint=1,
                     gated=gated,
                     resblocks=resblocks,
                     batchnorm=batchnorm,
                     save=save,
                     maxpool=maxpool,
                     activation=torch.nn.GELU,
                     init_func=torch.nn.init.kaiming_uniform_,
                     n_classes=1,
                     epochs_per_print=1,
                     size=size,
                     n_kernels=10
                     )
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "mom_range", "type": "choice", "values": [0, 0]},
            {"name": "niter", "type": "choice", "values": [1000, 1000]},
            {"name": "n_res", "type": "range", "bounds": [2, 10]},
            {"name": "scheduler", "type": "choice", "values":
                ['CycleScheduler', 'CycleScheduler']},
            {"name": "optimizer", "type": "choice", "values": ['adamw', 'adamw']},
            {"name": "weight_decay", "type": "range", "bounds": [1e-14, 1e-1], "log_scale": True},
            {"name": "momentum", "type": "range", "bounds": [0.9, 1.]},
            {"name": "learning_rate", "type": "range", "bounds": [1e-2, 1e-1], "log_scale": True},
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
