import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from fmri.utils.CycleAnnealScheduler import CycleScheduler
from fmri.utils.dataset import load_checkpoint, save_checkpoint, MRIDatasetClassifier, CTDataset, MRIDataset3D
from fmri.utils.transform_3d import Normalize

from fmri.models.unsupervised.SylvesterVAE3DCNN import SylvesterVAE
from fmri.utils.plot_performance import plot_performance
import torchvision
from torchvision import transforms
from ax.service.managed_loop import optimize
import random
from fmri.utils.activations import Swish
from tqdm import tqdm
import os

output_directory = "checkpoints"
import nibabel as nib
from fmri.utils.utils import validation_spliter


def random_init(m, init_func=torch.nn.init.xavier_uniform_):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        init_func(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


class Linear(nn.Module):
    def __init__(self,
                 n_neurons,
                 n_classes
                 ):
        super().__init__()
        self.dense1 = torch.nn.Linear(in_features=n_neurons, out_features=n_classes)

    def random_init(self, init_method=nn.init.xavier_uniform_):
        print("Random init")
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                init_method(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        return x

    def get_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def get_total_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)



class Train:
    def __init__(self,
                 input_size,
                 save,
                 images_path,
                 targets_path,
                 init_func=torch.nn.init.kaiming_uniform_,
                 activation=torch.nn.GELU,
                 batch_size=8,
                 epochs=1000,
                 fp16_run=False,
                 checkpoint_path=None,
                 epochs_per_checkpoint=1,
                 epochs_per_print=1,
                 gated=False,
                 batchnorm=False,
                 flow_type='vanilla',
                 verbose=2,
                 size=32,
                 mean=0.5,
                 std=0.5,
                 plot_perform=True,
                 val_share=0.1,
                 mode='valid',
                 early_stop=500,
                 cv=3,
                 load=False,
                 save_checkpoints=False,
                 n_embed=2048
                 ):
        super().__init__()
        self.input_size = input_size
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
        self.flow_type = flow_type
        self.save = save
        self.save_checkpoints = save_checkpoints
        self.load = load
        self.verbose = verbose
        self.images_path = images_path
        self.targets_path = targets_path
        self.size = size
        self.std = std
        self.mean = mean
        self.activation = activation
        self.init_func = init_func
        self.val_share = val_share
        self.plot_perform = plot_perform
        self.mode = mode
        self.early_stop = early_stop
        self.cross_validation = cv
        self.n_embed = n_embed
        self.model_name = ''

    def train(self, params):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        best_losses = []

        num_elements = params['num_elements']
        mom_range = params['mom_range']
        n_res = params['n_res']
        niter = params['niter']
        scheduler = params['scheduler']
        optimizer_type = params['optimizer']
        momentum = params['momentum']
        z_dim = params['z_dim']
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
        l1 = 0
        l2 = 0
        if self.verbose > 1:
            print("Parameters: \n\t",
                  'zdim: ' + str(z_dim) + "\n\t",
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

        if self.flow_type != 'o-sylvester':
            model = Linear(self.input_size, self.input_size).to(device)
        else:
            model = SylvesterVAE(z_dim=z_dim,
                                 maxpool=self.maxpool,
                                 input_size=self.input_size,
                                 out_channels=self.out_channels,
                                 kernel_sizes=self.kernel_sizes,
                                 kernel_sizes_deconv=self.kernel_sizes_deconv,
                                 strides=self.strides,
                                 strides_deconv=self.strides_deconv,
                                 dilatations=self.dilatations,
                                 dilatations_deconv=self.dilatations_deconv,
                                 padding=self.padding,
                                 padding_deconv=self.padding_deconv,
                                 batchnorm=self.batchnorm,
                                 flow_type=self.flow_type,
                                 n_res=n_res,
                                 gated=self.gated,
                                 has_dense=self.has_dense,
                                 resblocks=self.resblocks,
                                 h_last=z_dim,
                                 n_flows=n_flows,
                                 num_elements=num_elements,
                                 auxiliary=False,
                                 a_dim=0,

                                 )
        model.random_init()
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(params=model.parameters(),
                                         lr=learning_rate,
                                         weight_decay=weight_decay
                                         )
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

        # Load checkpoint if one exists
        epoch = 0
        best_loss = -1
        if self.checkpoint_path is not None and self.load:
            model, _, \
            epoch, losses, \
            kl_divs, losses_recon, \
            best_loss = load_checkpoint(checkpoint_path,
                                        model,
                                        self.maxpool,
                                        save=self.save,
                                        padding=self.padding,
                                        has_dense=self.has_dense,
                                        batchnorm=self.batchnorm,
                                        flow_type=self.flow_type,
                                        padding_deconv=self.padding_deconv,
                                        optimizer=optimizer,
                                        z_dim=z_dim,
                                        gated=self.gated,
                                        input_size=self.input_size,
                                        out_channels=self.out_channels,
                                        kernel_sizes=self.kernel_sizes,
                                        kernel_sizes_deconv=self.kernel_sizes_deconv,
                                        strides=self.strides,
                                        strides_deconv=self.strides_deconv,
                                        dilatations=self.dilatations,
                                        dilatations_deconv=self.dilatations_deconv,
                                        name=self.model_name,
                                        n_flows=n_flows,
                                        n_res=n_res,
                                        resblocks=resblocks,
                                        h_last=self.out_channels[-1],
                                        n_elements=num_elements
                                        )
        model = model.to(device)

        train_norm = transforms.Compose([
            torchvision.transforms.Normalize(mean=self.mean, std=self.std),
            Normalize()
        ])
        all_set = MRIDataset3D(self.images_path, self.targets_path, normalize=train_norm, transform=False)
        spliter = validation_spliter(all_set, cv=self.cross_validation)

        epoch_offset = max(1, epoch)

        # for cv in range(self.cross_validation):
        for cv in range(1):
            model.random_init()
            best_loss = -1
            valid_set, train_set = spliter.__next__()

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

            if optimizer_type == 'adam':
                optimizer = torch.optim.Adam(params=model.parameters(),
                                             lr=learning_rate,
                                             weight_decay=weight_decay
                                             )
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

            # Get shared output_directory ready
            os.makedirs('logs/' + self.model_name, exist_ok=True)
            self.model_name = '/'.join([
                "vae_3dcnn",
                'flows' + self.flow_type + str(n_flows),
                'bn' + str(self.batchnorm),
                'niter' + str(niter),
                'nres' + str(n_res),
                'momrange' + str(mom_range),
                'momentum' + str(momentum),
                str(optimizer_type),
                'zdim' + str(z_dim),
                'gated' + str(self.gated),
                'resblocks' + str(self.resblocks),
                'initlr' + str(learning_rate),
                'warmup' + str(warmup),
                'wd' + str(weight_decay),
                'l1' + str(l1),
                'l2' + str(l2),
                'size' + str(self.size),
            ])
            logger = SummaryWriter('logs/' + self.model_name)

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
                                             n_iter=self.epochs * len(train_loader),
                                             momentum=[
                                                 max(0.0, momentum - mom_range),
                                                 min(1.0, momentum + mom_range),
                                             ])

            losses = {
                "train": [],
                "valid": [],
            }
            kl_divs = {
                "train": [],
                "valid": [],
            }
            losses_recon = {
                "train": [],
                "valid": [],
            }
            running_abs_error = {
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
            valid_losses = []
            valid_kld = []
            valid_recons = []
            valid_abs_error = []
            train_losses = []
            train_abs_error = []
            train_kld = []
            train_recons = []
            best_epoch = False

            for epoch in range(epoch_offset, self.epochs):
                if early_stop_counter == self.early_stop:
                    if self.verbose > 0:
                        print('EARLY STOPPING.')
                    break
                model.train()
                # pbar = tqdm(total=len(train_loader))

                for i, batch in enumerate(train_loader):
                    # pbar.update(1)
                    model.zero_grad()
                    images, targets = batch
                    images = images.to(device)
                    targets = targets.to(device)
                    reconstruct, kl = model(images)
                    images = images.squeeze(1)
                    images = images.view(images.shape[0], -1)
                    loss_recon = criterion(
                        reconstruct,
                        targets.view(targets.shape[0], -1)
                    ).sum() / self.batch_size
                    kl_div = torch.mean(kl)
                    loss = loss_recon + kl_div
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

                    train_losses += [loss.item()]
                    train_kld += [kl_div.item()]
                    train_recons += [loss_recon.item()]
                    train_abs_error += [
                        float(torch.mean(torch.abs_(
                            reconstruct - images.to(device)
                        )).item())
                    ]

                    optimizer.step()
                    if scheduler == "CycleScheduler":
                        lr_schedule.step()
                        # optimizer = lr_schedule.optimizer
                    del kl, loss_recon, kl_div, loss

                target = nib.Nifti1Image(targets.detach().cpu().numpy()[0], np.eye(4))
                img = nib.Nifti1Image(images.detach().cpu().numpy()[0], np.eye(4))
                recon = nib.Nifti1Image(reconstruct.detach().cpu().numpy()[0], np.eye(4))
                if self.save and best_epoch:
                    if 'views' not in os.listdir():
                        os.mkdir('views')
                    target.to_filename(filename='views/Linear/target_train_.nii.gz')
                    img.to_filename(filename='views/Linear/image_train.nii.gz')
                    recon.to_filename(filename='views/Linear/reconstruct_train.nii.gz')

                losses["train"] += [np.mean(train_losses)]
                kl_divs["train"] += [np.mean(train_kld)]
                losses_recon["train"] += [np.mean(train_recons)]
                running_abs_error["train"] += [np.mean(train_abs_error)]
                logger.add_scalar('training_loss', np.mean(train_losses), epoch + 1)

                if epoch % self.epochs_per_print == 0:
                    if self.verbose > 1:
                        print("Epoch: {}:\t"
                              "Train Loss: {:.5f} , "
                              "kld: {:.3f} , "
                              "recon: {:.3f}"
                              .format(epoch,
                                      losses["train"][-1],
                                      kl_divs["train"][-1],
                                      losses_recon["train"][-1])
                              )
                    train_losses = []
                    train_abs_error = []
                    train_kld = []
                    train_recons = []

                model.eval()
                for i, batch in enumerate(valid_loader):
                    images, targets = batch
                    images = images.to(device)
                    targets = targets.to(device)
                    reconstruct, kl = model(images)
                    images = images.squeeze(1)
                    images = images.view(images.shape[0], -1)
                    loss_recon = criterion(
                        reconstruct,
                        targets.view(targets.shape[0], -1)
                    ).sum() / 2
                    kl_div = torch.mean(kl)
                    if epoch < warmup:
                        kl_div = kl_div * (epoch / warmup)
                    loss = loss_recon + kl_div
                    valid_losses += [loss.item()]
                    valid_kld += [kl_div.item()]
                    valid_recons += [loss_recon.item()]
                    valid_abs_error += [float(torch.mean(torch.abs_(reconstruct - images.to(device))).item())]
                logger.add_scalar('valid', np.mean(valid_losses), epoch + 1)
                losses["valid"] += [np.mean(valid_losses)]
                kl_divs["valid"] += [np.mean(valid_kld)]
                losses_recon["valid"] += [np.mean(valid_recons)]
                running_abs_error["valid"] += [np.mean(valid_abs_error)]
                if scheduler == "ReduceLROnPlateau":
                    if epoch - epoch_offset > 5:
                        lr_schedule.step(losses["valid"][-1])
                if (losses[self.mode][-1] < best_loss or best_loss == -1) and not np.isnan(losses[self.mode][-1]):
                    if self.verbose > 1:
                        print('BEST EPOCH!', losses[self.mode][-1])
                    early_stop_counter = 0
                    best_loss = losses[self.mode][-1]
                    best_epoch = True
                else:
                    early_stop_counter += 1

                if epoch % self.epochs_per_checkpoint == 0:
                    if self.save and best_epoch:
                        target = nib.Nifti1Image(targets.detach().cpu().numpy()[0], np.eye(4))
                        img = nib.Nifti1Image(images.detach().cpu().numpy()[0], np.eye(4))
                        recon = nib.Nifti1Image(reconstruct.detach().cpu().numpy()[0], np.eye(4))
                        target.to_filename(filename='views/Linear/target_valid.nii.gz')
                        img.to_filename(filename='views/Linear/image_valid.nii.gz')
                        recon.to_filename(filename='views/Linear/reconstruct_valid.nii.gz')
                    if best_epoch and self.save_checkpoints:
                        if self.verbose > 1:
                            print('Saving model...')
                        save_checkpoint(model=model,
                                        optimizer=optimizer,
                                        maxpool=None,
                                        padding=None,
                                        padding_deconv=None,
                                        learning_rate=learning_rate,
                                        epoch=epoch,
                                        checkpoint_path=output_directory,
                                        z_dim=z_dim,
                                        gated=self.gated,
                                        batchnorm=self.batchnorm,
                                        losses=losses,
                                        kl_divs=kl_divs,
                                        losses_recon=losses_recon,
                                        in_channels=None,
                                        out_channels=None,
                                        kernel_sizes=None,
                                        kernel_sizes_deconv=None,
                                        strides=None,
                                        strides_deconv=None,
                                        dilatations=None,
                                        dilatations_deconv=None,
                                        best_loss=best_loss,
                                        save=self.save,
                                        name=self.model_name,
                                        n_flows=n_flows,
                                        flow_type=self.flow_type,
                                        n_res=n_res,
                                        resblocks=resblocks,
                                        h_last=z_dim,
                                        n_elements=num_elements,
                                        n_kernels=None
                                        )
                if epoch % self.epochs_per_print == 0:
                    if self.verbose > 0:
                        print("Epoch: {}:\t"
                              "Valid Loss: {:.5f} , "
                              "kld: {:.3f} , "
                              "recon: {:.3f}"
                              .format(epoch,
                                      losses["valid"][-1],
                                      kl_divs["valid"][-1],
                                      losses_recon["valid"][-1]
                                      )
                              )
                        valid_losses = []
                        valid_kld = []
                        valid_recons = []
                        valid_abs_error = []

                    if self.verbose > 1:
                        print("Current LR:", optimizer.param_groups[0]['lr'])
                    if 'momentum' in optimizer.param_groups[0].keys():
                        print("Current Momentum:", optimizer.param_groups[0]['momentum'])
                if self.plot_perform:
                    plot_performance(loss_total=losses, losses_recon=losses_recon, kl_divs=kl_divs, shapes=shapes,
                                     results_path="../figures",
                                     filename="training_loss_trace_"
                                              + self.model_name + '.jpg')
            if self.verbose > 0:
                print('BEST LOSS :', best_loss)
            best_losses += [best_loss]
        return min(best_losses)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    torch.manual_seed(42)

    random.seed(42)

    size = 32
    z_dim = 500
    input_size = [14*14*14, ]
    n_flows = 10
    bs = 16
    flow_type = 'vanilla'
    epochs_per_checkpoint = 1
    has_dense = True
    batchnorm = True
    gated = False
    resblocks = True
    checkpoint_path = "checkpoints"
    images_path = 'D:\\workbench\\projects\\AutoTKV_MouseMRI-master\\AllTrainingImages\\images\\'
    targets_path = 'D:\\workbench\\projects\\AutoTKV_MouseMRI-master\\AllTrainingImages\\targets\\'

    n_epochs = 10000
    save = True
    training = Train(input_size=input_size,
                     images_path=images_path,
                     targets_path=targets_path,
                     batch_size=bs,
                     epochs=n_epochs,
                     checkpoint_path=checkpoint_path,
                     epochs_per_checkpoint=epochs_per_checkpoint,
                     gated=gated,
                     fp16_run=False,
                     batchnorm=batchnorm,
                     flow_type=flow_type,
                     save=save,
                     plot_perform=False,
                     mean=0.1,
                     std=0.1,
                     init_func=torch.nn.init.xavier_uniform_,
                     mode='valid',
                     load=False,
                     cv=5
                     )

    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "warmup", "type": "choice", "values": [0, 0]},
            {"name": "mom_range", "type": "range", "bounds": [0., 0.1]},
            {"name": "num_elements", "type": "choice", "values": [2, 3]},
            {"name": "niter", "type": "choice", "values": [0, 0]},
            {"name": "n_res", "type": "range", "bounds": [1, 2]},
            {"name": "z_dim", "type": "choice", "values": [32, 32]},
            {"name": "n_flows", "type": "choice", "values": [10, 10]},
            {"name": "scheduler", "type": "choice", "values":
                ['ReduceLROnPlateau', 'ReduceLROnPlateau']},
            {"name": "optimizer", "type": "choice", "values": ['adam', 'adam']},
            {"name": "l1", "type": "range", "bounds": [1e-8, 1e-1], "log_scale": True},
            {"name": "l2", "type": "range", "bounds": [1e-8, 1e-1], "log_scale": True},
            {"name": "weight_decay", "type": "range", "bounds": [1e-8, 1e-5], "log_scale": True},
            {"name": "momentum", "type": "choice", "values": [0, 0]},
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
