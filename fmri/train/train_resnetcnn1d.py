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
from fmri.utils.transform_3d import Normalize, Flip90, Flip180, Flip270, XFlip, YFlip, ZFlip
from fmri.models.supervised.resnetcnn3d import ConvResnet3D
from fmri.utils.plot_performance import plot_performance
import torchvision
from torchvision import transforms
from ax.service.managed_loop import optimize
import random

import os

import nibabel as nib
from fmri.utils.utils import validation_split

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = "cpu"


class Train:
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 strides,
                 dilatations,
                 save,
                 padding,
                 path,
                 n_classes,
                 init_func=torch.nn.init.kaiming_uniform_,
                 activation=torch.nn.GELU,
                 batch_size=8,
                 epochs=1000,
                 fp16_run=False,
                 checkpoint_path=None,
                 epochs_per_checkpoint=1,
                 epochs_per_print=10,
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
                 val_share=0.1
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

    def train(self, params):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        num_elements = params['num_elements']
        mom_range = params['mom_range']
        n_res = params['n_res']
        niter = params['niter']
        scheduler = params['scheduler']
        optimizer_type = params['optimizer']
        momentum = params['momentum']
        learning_rate = params['learning_rate'].__format__('e')
        n_flows = params['n_flows']
        weight_decay = params['weight_decay'].__format__('e')
        warmup = params['warmup']
        l1 = params['l1'].__format__('e')
        l2 = params['l2'].__format__('e')

        weight_decay = float(str(weight_decay)[:1] + str(weight_decay)[-4:])
        learning_rate = float(str(learning_rate)[:1] + str(learning_rate)[-4:])
        l1 = float(str(l1)[:1] + str(l1)[-4:])
        l2 = float(str(l2)[:1] + str(l2)[-4:])
        if self.verbose > 1:
            print("Parameters: \n\t",
                  'zdim: ' + str(self.n_classes) + "\n\t",
                  'mom_range: ' + str(mom_range) + "\n\t",
                  'num_elements: ' + str(num_elements) + "\n\t",
                  'niter: ' + str(niter) + "\n\t",
                  'nres: ' + str(n_res) + "\n\t",
                  'learning_rate: ' + learning_rate.__format__('e') + "\n\t",
                  'momentum: ' + str(momentum) + "\n\t",
                  'n_flows: ' + str(n_flows) + "\n\t",
                  'weight_decay: ' + weight_decay.__format__('e') + "\n\t",
                  'warmup: ' + str(warmup) + "\n\t",
                  'l1: ' + l1.__format__('e') + "\n\t",
                  'l2: ' + l2.__format__('e') + "\n\t",
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
                         + '_warmup' + str(warmup) \
                         + '_wd' + weight_decay.__format__('e') \
                         + '_l1' + l1.__format__('e') \
                         + '_l2' + l2.__format__('e') \
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
                             activation=torch.nn.ReLU,
                             n_res=n_res,
                             gated=self.gated,
                             has_dense=self.has_dense,
                             resblocks=self.resblocks,
                             ).to(device)
        model.random_init()
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
            best_loss = load_checkpoint(checkpoint_path,
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
                                        n_flows=n_flows,
                                        n_res=n_res,
                                        resblocks=resblocks,
                                        h_last=None,
                                        n_elements=None
                                        )
        model = model.to(device)
        # t1 = torch.Tensor(np.load('/run/media/simon/DATA&STUFF/data/biology/arrays/t1.npy'))
        # targets = torch.Tensor([0 for _ in t1])

        train_transform = transforms.Compose([
            XFlip(),
            YFlip(),
            ZFlip(),
            Flip90(),
            Flip180(),
            Flip270(),
            torchvision.transforms.Normalize(mean=(self.mean), std=(self.std)),
            Normalize()
        ])
        all_set = MRIDatasetClassifier(self.path, transform=train_transform, size=self.size)
        train_set, valid_set = validation_split(all_set, val_share=self.val_share)

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
        epoch_offset = max(1, epoch)

        if scheduler == 'ReduceLROnPlateau':
            lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                     factor=0.1,
                                                                     cooldown=50,
                                                                     patience=200,
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
        print("Training Started on device:", device)
        for epoch in range(epoch_offset, self.epochs):
            if early_stop_counter == 500:
                if self.verbose > 0:
                    print('EARLY STOPPING.')
                break
            best_epoch = False
            model.train()
            train_losses = []
            train_accuracy = []

            # pbar = tqdm(total=len(train_loader))
            for i, batch in enumerate(train_loader):
                #    pbar.update(1)
                model.zero_grad()
                images, targets = batch
                images = torch.autograd.Variable(images).to(device)
                targets = torch.autograd.Variable(targets).to(device)
                # images = images.unsqueeze(1)
                preds = model(images)
                images = images.squeeze(1)

                loss = criterion(preds, targets)
                l2_reg = torch.tensor(0.)
                l1_reg = torch.tensor(0.)
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        l1_reg = l1 + torch.norm(param, 1)
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        l2_reg = l2 + torch.norm(param, 1)
                loss += l1 * l1_reg
                loss += l2 * l2_reg

                loss.backward()
                accuracy = sum([1 if torch.argmax(pred) == target else 0 for (pred, target) in zip(preds, targets)]) / len(targets)
                train_accuracy += [accuracy]

                train_losses += [loss.item()]

                optimizer.step()
                logger.add_scalar('training_loss', loss.item(), i + len(train_loader) * epoch)
                del loss

            losses["train"] += [np.mean(train_losses)]
            accuracies["train"] += [np.mean(train_accuracy)]

            if epoch % self.epochs_per_print == 0:
                if self.verbose > 1:
                    print("Epoch: {}:\t"
                          "Train Loss: {:.5f} , "
                          "Accuracy: {:.3f} , "
                          .format(epoch,
                                  losses["train"][-1],
                                  accuracies["train"][-1]
                                  ))

            model.eval()
            valid_losses = []
            valid_accuracy = []
            # pbar = tqdm(total=len(valid_loader))
            for i, batch in enumerate(valid_loader):
                #    pbar.update(1)
                images, targets = batch
                images = torch.autograd.Variable(images).to(device)
                targets = torch.autograd.Variable(targets).to(device)
                preds = model(images)

                loss = criterion(preds, targets)
                valid_losses += [loss.item()]
                accuracy = sum([1 if torch.argmax(pred) == target else 0 for (pred, target) in zip(preds, targets)]) / len(targets)
                valid_accuracy += [accuracy]
                logger.add_scalar('training loss', np.log2(loss.item()), i + len(train_loader) * epoch)
            losses["valid"] += [np.mean(valid_losses)]
            accuracies["valid"] += [np.mean(valid_accuracy)]
            if epoch - epoch_offset > 5:
                lr_schedule.step(losses["valid"][-1])
            # should be valid, but train is ok to test if it can be done without caring about
            # generalisation
            mode = 'valid'
            if (losses[mode][-1] < best_loss or best_loss == -1) and not np.isnan(losses[mode][-1]):
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
                                    checkpoint_path=None,
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
                                    n_elements=None
                                    )
            if epoch % self.epochs_per_print == 0:
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
            #if self.plot_perform:
            #    plot_performance(loss_total=losses, losses_recon=losses_recon, kl_divs=kl_divs, shapes=shapes,
            #                     results_path="../figures",
            #                     filename="training_loss_trace_"
            #                              + self.modelname + '.jpg')
        if self.verbose > 0:
            print('BEST LOSS :', best_loss)
        return best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.manual_seed(11)

    random.seed(10)

    size = 32
    in_channels = [1, 32, 64, 128, 256]
    out_channels = [32, 64, 128, 256, 256]
    kernel_sizes = [3, 3, 3, 3, 3]
    kernel_sizes_deconv = [3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1]
    strides_deconv = [1, 1, 1, 1, 1]
    dilatations = [1, 1, 1, 1, 1]
    dilatations_Deconv = [1, 1, 1, 1, 1, 1]
    paddings = [1, 1, 1, 1, 1]
    paddings_deconv = [1, 1, 1, 1, 1]
    dilatations_deconv = [1, 1, 1, 1, 1]
    n_flows = 10
    bs = 8
    maxpool = 2
    has_dense = True
    batchnorm = True
    gated = False
    resblocks = True
    checkpoint_path = "checkpoints"
    path = '/Users/simonpelletier/Downloads/recordings/'

    n_epochs = 10000
    save = False
    training = Train(in_channels=in_channels,
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
                     activation=Swish,
                     init_func=torch.nn.init.kaiming_uniform_,
                     n_classes=2
                     )
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "warmup", "type": "choice", "values": [0, 0]},
            {"name": "mom_range", "type": "choice", "values": [0, 0]},
            {"name": "num_elements", "type": "range", "bounds": [1, 5]},
            {"name": "niter", "type": "choice", "values": [10, 10]},
            {"name": "n_res", "type": "range", "bounds": [0, 10]},
            {"name": "n_flows", "type": "range", "bounds": [2, 20]},
            {"name": "scheduler", "type": "choice", "values":
                ['ReduceLROnPlateau', 'ReduceLROnPlateau']},
            {"name": "optimizer", "type": "choice", "values": ['adamw', 'adamw']},
            {"name": "l1", "type": "range", "bounds": [1e-14, 1e-1], "log_scale": True},
            {"name": "l2", "type": "range", "bounds": [1e-14, 1e-1], "log_scale": True},
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
