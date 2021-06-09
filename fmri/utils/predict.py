from fmri.models.supervised.cnn3d import ConvResnet3D
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from scipy.stats import norm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from fmri.models.utils.distributions import log_gaussian
from fmri.utils.activations import Swish, Mish
from fmri.utils.CycleAnnealScheduler import CycleScheduler
from fmri.utils.dataset import load_checkpoint, save_checkpoint, MRIDatasetClassifier, CTDatasetInfere
from fmri.utils.transform_3d import Normalize, RandomRotation3D, ColorJitter3D, Flip90, Flip180, Flip270, XFlip, YFlip, \
    ZFlip, RandomAffine3D
from fmri.models.supervised.resnetcnn3d import ConvResnet3D
from fmri.utils.plot_performance import plot_performance
import torchvision
from torchvision import transforms
from ax.service.managed_loop import optimize
import random

import os

import nibabel as nib
from fmri.utils.utils import validation_spliter

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = "cpu"


class Predict:
    def __init__(self,
                 basedir,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 strides,
                 dilatations,
                 padding,
                 train_path,
                 test_path,
                 n_classes,
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
                 max_fvc=0,
                 n_kernels=1,
                 random_node='output',
                 train_labels_path='/run/media/simon/DATA&STUFF/data/train.csv',
                 test_labels_path='/run/media/simon/DATA&STUFF/data/test.csv',
                 submission_file='/run/media/simon/DATA&STUFF/data/sample_submission.csv'
                 ):
        super().__init__()
        self.max_fvc = max_fvc
        self.n_kernels = n_kernels
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
        self.verbose = verbose
        self.train_path = train_path
        self.test_path = test_path
        self.size = size
        self.std = std
        self.mean = mean
        self.activation = activation
        self.init_func = init_func
        self.val_share = val_share
        self.plot_perform = plot_perform
        self.cross_validation = cross_validation
        self.is_bayesian = is_bayesian
        self.train_labels_path = train_labels_path
        self.test_labels_path = test_labels_path
        self.submission_file = submission_file
        self.basedir = basedir

    def predict(self, params):
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
                             is_bayesian=self.is_bayesian,
                             activation=torch.nn.ReLU,
                             n_res=n_res,
                             gated=self.gated,
                             has_dense=self.has_dense,
                             resblocks=self.resblocks,
                             max_fvc=self.max_fvc,
                             n_kernels=self.n_kernels
                             ).to(device)
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
        model, optimizer, \
        epoch, losses, \
        kl_divs, losses_recon, \
        best_loss = load_checkpoint(self.checkpoint_path,
                                    model,
                                    self.maxpool,
                                    save=False,
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
                                    predict=True
                                    )
        model = model.to(device)

        test_set = CTDatasetInfere(train_path=self.train_path,
                                   test_path=self.test_path,
                                   train_labels_path=self.train_labels_path,
                                   test_labels_path=self.test_labels_path,
                                   submission_file=self.submission_file,
                                   size=self.size)
        test_loader = DataLoader(test_set,
                                 num_workers=0,
                                 shuffle=False,
                                 batch_size=1,
                                 pin_memory=False,
                                 drop_last=False)

        # pbar = tqdm(total=len(train_loader))
        f = open(self.basedir + "/submission.csv", "w")
        f.write("Patient_Week,FVC,Confidence\n")
        for i, batch in enumerate(test_loader):
            #    pbar.update(1)
            patient, images, targets, patient_info = batch
            patient_info = patient_info.to(device)

            images = images.to(device)
            targets = targets.to(device)

            _, mu, log_var = model(images, patient_info)

            l1_loss = l1(mu * test_set.max_fvc, targets.cuda())

            fvc = l1_loss.item()
            confidence = 2 * np.exp(np.sqrt(log_var.item())) * test_set.max_fvc
            f.write(",".join([patient[0], str(int(fvc)), str(int(confidence))]))
            f.write('\n')
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.manual_seed(11)

    random.seed(10)

    size = 32
    in_channels = [1, 256, 256, 512, 1024]
    out_channels = [256, 256, 512, 1024, 1024]
    kernel_sizes = [3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1]
    dilatations = [1, 1, 1, 1, 1]
    paddings = [1, 1, 1, 1, 1]
    bs = 8
    maxpool = 2
    has_dense = True
    batchnorm = True
    gated = False
    resblocks = False
    checkpoint_path = "../train/checkpoints"
    train_path = '/run/media/simon/DATA&STUFF/data/train_32x32/'
    test_path = '/run/media/simon/DATA&STUFF/data/test_32x32/'
    basedir = '../../'
    params = {
        'mom_range': 0,
        'n_res': 0,
        'niter': 1000,
        'scheduler': 'CycleScheduler',
        'optimizer': 'adamw',
        'momentum': 0.9723667767830193,
        'learning_rate': 4.000000e-04,
        'weight_decay': 1.000000e-05,
    }
    predict = Predict(basedir,
                      in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_sizes=kernel_sizes,
                      strides=strides,
                      dilatations=dilatations,
                      train_path=train_path,
                      test_path=test_path,
                      padding=paddings,
                      batch_size=bs,
                      checkpoint_path=checkpoint_path,
                      epochs_per_checkpoint=1,
                      gated=gated,
                      resblocks=resblocks,
                      batchnorm=batchnorm,
                      maxpool=maxpool,
                      activation=torch.nn.ReLU,
                      init_func=torch.nn.init.xavier_uniform_,
                      n_classes=1,
                      epochs_per_print=10,
                      size=size
                      )
    from matplotlib import pyplot as plt

    predict.predict(params)
