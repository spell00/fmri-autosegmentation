import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from fmri.utils.CycleAnnealScheduler import CycleScheduler
from fmri.utils.dataset import load_checkpoint, save_checkpoint, MRIDataset3D
from fmri.models.unsupervised.VAE_2DCNN import Autoencoder2DCNN
from fmri.models.unsupervised.VQVAE2_3D import VQVAE

from fmri.utils.plot_performance import plot_performance
import torchvision
from torchvision import transforms
from ax.service.managed_loop import optimize
import random
from fmri.models.unsupervised.keras.metrics import jaccard_distance_loss, dice_coef_loss
from fmri.utils.activations import Swish
from tqdm import tqdm
import os

output_directory = "checkpoints"
import nibabel as nib
from fmri.utils.utils import validation_spliter

# source: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
# PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, y_true, y_preds, smooth=1):
        inputs = y_preds.view(-1)
        targets = y_true.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


# PyTorch
class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(JaccardLoss, self).__init__()

    def forward(self, y_true, y_preds, smooth=1):
        inputs = y_preds.view(-1)
        targets = y_true.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        jaccard = (intersection + smooth) / (union + smooth)

        return 1 - jaccard


class MCCLoss(nn.Module):
    def __init__(self):
        super(MCCLoss, self).__init__()

dice_loss = DiceLoss()
jaccard_loss = JaccardLoss()

class Train:
    def __init__(self,
                 in_channels,
                 in_channels2,
                 out_channels,
                 out_channels2,
                 kernel_sizes,
                 kernel_sizes_deconv,
                 strides,
                 strides_deconv,
                 dilatations,
                 dilatations_deconv,
                 save,
                 padding,
                 padding2,
                 padding_deconv,
                 padding_deconv2,
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
                 has_dense=True,
                 batchnorm=False,
                 resblocks=False,
                 flow_type='vanilla',
                 maxpool=2,
                 maxpool2=3,
                 verbose=2,
                 size=32,
                 mean=0.5,
                 std=0.5,
                 plot_perform=True,
                 val_share=0.1,
                 mode='valid',
                 early_stop=250,
                 cv=3,
                 load=False,
                 save_checkpoints=False,
                 n_embed=2048,
                 scheduler='CycleScheduler',
                 optimizer_type='adam'
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.in_channels2 = in_channels2
        self.out_channels = out_channels
        self.out_channels2 = out_channels2
        self.kernel_sizes = kernel_sizes
        self.kernel_sizes_deconv = kernel_sizes_deconv
        self.strides = strides
        self.strides_deconv = strides_deconv
        self.dilatations = dilatations
        self.dilatations_deconv = dilatations_deconv
        self.padding = padding
        self.padding2 = padding2
        self.padding_deconv = padding_deconv
        self.padding_deconv2 = padding_deconv
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
        self.maxpool = maxpool
        self.maxpool2 = maxpool2
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
        self.scheduler = scheduler
        self.optimizer_type = optimizer_type
        self.lr_schedule = None

    def compute_confusion_matrix(self, y_test, y_classes):
        ys = torch.round(y_test.reshape([y_test.shape[0], -1]))
        y_classes = y_classes.reshape([y_classes.shape[0], -1])
        tp = np.sum([[1 if true == 1 and pred == 1 else 0 for true, pred in zip(t, c)] for t, c in zip(ys, y_classes)])
        tn = np.sum([[1 if true == 0 and pred == 0 else 0 for true, pred in zip(t, c)] for t, c in zip(ys, y_classes)])
        fp = np.sum([[1 if true == 0 and pred == 1 else 0 for true, pred in zip(t, c)] for t, c in zip(ys, y_classes)])
        fn = np.sum([[1 if true == 1 and pred == 0 else 0 for true, pred in zip(t, c)] for t, c in zip(ys, y_classes)])

        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        # jaccard = jaccard_loss(inputs=y_classes, targets=y_test)
        # dice = dice_loss(inputs=y_classes, targets=y_test)

        return sensitivity, specificity # , jaccard, dice


    def train(self, params):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        best_losses = []

        # n_res = params['n_res']
        z_dim = params['z_dim']
        learning_rate = params['learning_rate'].__format__('e')
        # n_flows = params['n_flows']
        weight_decay = params['weight_decay'].__format__('e')
        # l1 = params['l1'].__format__('e')

        weight_decay = float(str(weight_decay)[:1] + str(weight_decay)[-4:])
        learning_rate = float(str(learning_rate)[:1] + str(learning_rate)[-4:])
        # l1 = float(str(l1)[:1] + str(l1)[-4:])
        if self.verbose > 1:
            print("Parameters: \n\t",
                  'zdim: ' + str(z_dim) + "\n\t",
                  # 'nres: ' + str(n_res) + "\n\t",
                  'learning_rate: ' + learning_rate.__format__('e') + "\n\t",
                  # 'n_flows: ' + str(n_flows) + "\n\t",
                  'weight_decay: ' + weight_decay.__format__('e') + "\n\t",
                  # 'l1: ' + str(l1) + "\n\t",
                  'optimizer_type: ' + self.optimizer_type + "\n\t",
                  'in_channels:' + "-".join([str(item) for item in self.in_channels]) + "\n\t",
                  'out_channels:' + "-".join([str(item) for item in self.out_channels]) + "\n\t",
                  'kernel_sizes:' + "-".join([str(item) for item in self.kernel_sizes]) + "\n\t",
                  'kernel_sizes_deconv:' + "-".join([str(item) for item in self.kernel_sizes_deconv]) + "\n\t",
                  'paddings:' + "-".join([str(item) for item in self.padding]) + "\n\t",
                  'padding_deconv:' + "-".join([str(item) for item in self.padding_deconv]) + "\n\t",
                  'dilatations:' + "-".join([str(item) for item in self.dilatations]) + "\n\t",
                  'dilatations_deconv:' + "-".join([str(item) for item in self.dilatations_deconv]) + "\n\t",
                  )
        '''
        if self.flow_type != 'o-sylvester':
            model = Autoencoder3DCNN(z_dim,
                                     self.max_pool,
                                     self.in_channels,
                                     self.out_channels,
                                     self.kernel_sizes,
                                     self.kernel_sizes_deconv,
                                     self.strides,
                                     self.strides_deconv,
                                     self.dilatations,
                                     self.dilatations_deconv,
                                     self.padding,
                                     self.padding_deconv,
                                     has_dense=self.has_dense,
                                     batchnorm=self.batchnorm,
                                     flow_type=self.flow_type,
                                     n_flows=n_flows,
                                     n_res=n_res,
                                     gated=self.gated,
                                     resblocks=self.resblocks
                                     ).to(device)
        else:
            model = SylvesterVAE(z_dim=z_dim,
                                 max_pool=self.max_pool,
                                 in_channels=self.in_channels,
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
        '''
        model = VQVAE(
            in_channel=1,
            channel=512,
            n_res_block=5,
            n_res_channel=64,
            embed_dim=z_dim,
            n_embed=1024,
        )
        model.random_init()
        criterion = nn.BCELoss(reduction="none")
        # criterion = DiceLoss()
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(params=model.parameters(),
                                         lr=learning_rate,
                                         weight_decay=weight_decay
                                         )
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(params=model.parameters(),
                                        lr=learning_rate,
                                        weight_decay=weight_decay,
                                        momentum=0.9)
        elif self.optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(params=model.parameters(),
                                            lr=learning_rate,
                                            weight_decay=weight_decay,
                                            momentum=0.9)
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
                                        in_channels=self.in_channels,
                                        out_channels=self.out_channels,
                                        kernel_sizes=self.kernel_sizes,
                                        kernel_sizes_deconv=self.kernel_sizes_deconv,
                                        strides=self.strides,
                                        strides_deconv=self.strides_deconv,
                                        dilatations=self.dilatations,
                                        dilatations_deconv=self.dilatations_deconv,
                                        name=self.model_name,
                                        n_flows=None,
                                        n_res=5,
                                        resblocks=resblocks,
                                        h_last=self.out_channels[-1],
                                        n_elements=None
                                        )
        model = model.to(device)

        train_norm = transforms.Compose([
            torchvision.transforms.Normalize(mean=self.mean, std=self.std),
        ])
        all_set = MRIDataset3D(self.images_path, self.targets_path, size=self.size, normalize=train_norm, transform=False)
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
                                      batch_size=1,
                                      pin_memory=False,
                                      drop_last=True)

            # Get shared output_directory ready
            self.model_name = '/'.join([
                f"vae_2d_cnn_{self.optimizer_type}",
                f'n_res{5}',
                f'z_dim{z_dim}',
                f'wd{weight_decay}',
                # f'l1{l1}'
            ])
            os.makedirs('logs/' + self.model_name, exist_ok=True)
            logger = SummaryWriter('logs/' + self.model_name)
            # logger.add_graph(model)
            if self.scheduler == 'ReduceLROnPlateau':
                lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                         factor=0.1,
                                                                         cooldown=50,
                                                                         patience=100,
                                                                         verbose=True,
                                                                         min_lr=1e-15)
            elif self.scheduler == 'CycleScheduler':
                lr_schedule = CycleScheduler(optimizer,
                                             learning_rate,
                                             n_iter=self.epochs * len(train_loader))

            results = {
                "losses": {
                    "train": [],
                    "valid": [],
                },
                "kl_divs": {
                    "train": [],
                    "valid": [],
                },
                "recon": {
                    "train": [],
                    "valid": [],
                },
                "abs_error": {
                    "train": [],
                    "valid": [],
                },
                "jaccard": {
                    "train": [],
                    "valid": [],
                },
                "dice": {
                    "train": [],
                    "valid": [],
                },
                "acc": {
                    "train": [],
                    "valid": [],
                },
                "voxel_rel_diff": {
                    "train": [],
                    "valid": [],
                }

            }
            shapes = {
                "train": len(train_set),
                "valid": len(valid_set),
            }
            early_stop_counter = 0
            print("\n\n\nCV:", cv, "/", self.cross_validation, "\nTrain samples:", len(train_set),
                  "\nValid samples:", len(valid_set), "\n\n\n")

            is_best_epoch = False
            best_epoch = 0
            for epoch in range(epoch_offset, self.epochs):
                traces = {
                    "losses": {
                        "train": [],
                        "valid": [],
                    },
                    "kl_divs": {
                        "train": [],
                        "valid": [],
                    },
                    "recon": {
                        "train": [],
                        "valid": [],
                    },
                    "abs_error": {
                        "train": [],
                        "valid": [],
                    },
                    "acc": {
                        "train": [],
                        "valid": [],
                    },
                    "voxel_rel_diff": {
                        "train": [],
                        "valid": [],
                    },
                    "jaccard": {
                        "train": [],
                        "valid": [],
                    },
                    "dice": {
                        "train": [],
                        "valid": [],
                    }
                }
                if early_stop_counter == self.early_stop:
                    if self.verbose > 0:
                        print('EARLY STOPPING.')
                    break
                model.train()
                # pbar = tqdm(total=len(train_loader))

                for i, batch in enumerate(train_loader):
                    # pbar.update(1)
                    model.zero_grad()
                    images, targets, voxel_count, voxels_ratio = batch
                    images = images.to(device)
                    targets = targets.to(device)
                    reconstruct, kl = model(images)
                    bin_reconstruct = torch.round(reconstruct)
                    # reconstruct = reconstruct.squeeze(1)
                    # images = images.squeeze(1)
                    # targets = targets.squeeze(1)
                    loss_recon = criterion(
                        reconstruct,
                        targets
                    ).sum() / self.batch_size
                    kl_div = torch.mean(kl)
                    loss = loss_recon + kl_div
                    # l2_reg = torch.tensor(0.)
                    # l1_reg = torch.tensor(0.)
                    # for name, param in model.named_parameters():
                    #    if 'weight' in name:
                    #         l1_reg = l1 + torch.norm(param, 1)
                    # loss += l1 * l1_reg
                    loss.backward()

                    total = len(targets[0].view(-1))
                    acc = np.mean([torch.sum(x == y).item() for x, y in zip(bin_reconstruct, targets)]) / total
                    jaccard = jaccard_loss(targets, bin_reconstruct)
                    dice = dice_loss(targets, bin_reconstruct)

                    voxel_rel_diff = torch.mean((torch.sum(bin_reconstruct.cpu().view(self.batch_size, -1), 1) / voxels_ratio - voxel_count) / voxel_count)

                    traces['acc']['train'] += [acc]
                    traces['voxel_rel_diff']['train'] += [voxel_rel_diff.item()]
                    traces['losses']['train'] += [loss.item()]
                    traces['kl_divs']['train'] += [kl_div.item()]
                    traces['recon']['train'] += [loss_recon.item()]
                    traces['dice']['train'] += [dice.item()]
                    traces['jaccard']['train'] += [jaccard.item()]
                    traces['abs_error']['train'] += [
                        float(torch.mean(torch.abs_(
                            reconstruct - images.to(device)
                        )).item())
                    ]

                    optimizer.step()
                    if self.scheduler == "CycleScheduler":
                        lr_schedule.step()
                        # optimizer = lr_schedule.optimizer
                    del kl, loss_recon, kl_div, loss

                target = nib.Nifti1Image(targets.detach().cpu().numpy()[0], np.eye(4))
                img = nib.Nifti1Image(images.detach().cpu().numpy()[0], np.eye(4))
                recon = nib.Nifti1Image(reconstruct.detach().cpu().numpy()[0], np.eye(4))
                bin_recon = nib.Nifti1Image(bin_reconstruct.detach().cpu().numpy()[0], np.eye(4))
                if self.save and is_best_epoch:
                    if f'views/{self.model_name}' not in os.listdir():
                        os.makedirs(f'views/{self.model_name}', exist_ok=True)
                    target.to_filename(filename=f'views/{self.model_name}/target_train_.nii.gz')
                    img.to_filename(filename=f'views/{self.model_name}/image_train.nii.gz')
                    recon.to_filename(filename=f'views/{self.model_name}/reconstruct_train.nii.gz')
                    bin_recon.to_filename(filename=f'views/{self.model_name}/reconstruct_train_bin.nii.gz')
                    middle = 3
                    logger.add_image(tag='train/target', img_tensor=targets.detach().cpu()[0], global_step=epoch)
                    logger.add_image(tag='train/reconstruction', img_tensor=reconstruct.detach().cpu()[0], global_step=epoch)
                    logger.add_image(tag='train/binary_reconstruction', img_tensor=bin_reconstruct.detach().cpu()[0], global_step=epoch)
                    logger.add_image(tag='train/input', img_tensor=images.detach().cpu()[0], global_step=epoch)
                results["losses"]["train"] += [np.mean(traces['losses']['train'])]
                results["kl_divs"]["train"] += [np.mean(traces['kl_divs']['train'])]
                results["recon"]["train"] += [np.mean(traces['recon']['train'])]
                results["abs_error"]["train"] += [np.mean(traces['abs_error']['train'])]
                results["acc"]["train"] += [np.mean(traces['acc']['train'])]
                results["voxel_rel_diff"]["train"] += [np.mean(traces['voxel_rel_diff']['train'])]
                results["jaccard"]["train"] += [np.mean(traces['jaccard']['train'])]
                results["dice"]["train"] += [np.mean(traces['dice']['train'])]

                logger.add_scalar('train/loss', results["losses"]["train"][-1], epoch + 1)
                logger.add_scalar('train/kld', results["kl_divs"]["train"][-1], epoch + 1)
                logger.add_scalar('train/recon',  results["recon"]["train"][-1], epoch + 1)
                logger.add_scalar('train/accuracy', results["acc"]["train"][-1], epoch + 1)
                logger.add_scalar('train/jaccard', results["jaccard"]["train"][-1], epoch + 1)
                logger.add_scalar('train/dice', results["dice"]["train"][-1], epoch + 1)
                logger.add_scalar('train/voxel_rel_diff', results['voxel_rel_diff']['train'][-1], epoch + 1)

                if epoch % self.epochs_per_print == 0:
                    if self.verbose > 1:
                        print("Train Loss: {:.5f} ,  kld: {:.3f} ,  recon: {:.3f},  acc: {:.2f}".format(
                                      results['losses']["train"][-1],
                                      results['kl_divs']["train"][-1],
                                      results['losses']["train"][-1],
                                      results['acc']["train"][-1],
                                      ),
                            f"Jaccard: {np.round(results['jaccard']['train'][-1], 4)}, "
                            f"Dice: {np.round(results['dice']['train'][-1], 4)}, "
                            f"Voxel Rel Diff: {np.round(results['voxel_rel_diff']['train'][-1], 4)}"
                        )

                model.eval()
                for i, batch in enumerate(valid_loader):
                    images, targets, voxel_count, voxels_ratio = batch
                    images = images.to(device)
                    targets = targets.to(device)

                    reconstruct, kl = model(images)

                    bin_reconstruct = torch.round(reconstruct)
                    #  = reconstruct.squeeze(1)
                    # images = images.squeeze(1)
                    # targets = targets.squeeze(1)
                    # bin_reconstruct = bin_reconstruct.squeeze(1)
                    loss_recon = criterion(
                        reconstruct,
                        targets
                    ).sum()
                    kl_div = torch.mean(kl)
                    loss = loss_recon + kl_div
                    jaccard = jaccard_loss(
                        y_true=targets.reshape(targets.shape[0], -1),
                        y_preds=bin_reconstruct.reshape(targets.shape[0], -1)
                    )
                    dice = dice_loss(y_true=targets, y_preds=bin_reconstruct)

                    voxel_rel_diff = torch.mean((torch.sum(bin_reconstruct.cpu().view(bin_reconstruct.shape[0], -1), 1) / voxels_ratio - voxel_count) / voxel_count)

                    total = len(targets[0].view(-1))
                    acc = np.mean([torch.sum(x == y).item() for x, y in zip(bin_reconstruct, targets)]) / total
                    traces['acc']['valid'] += [acc]
                    traces['voxel_rel_diff']['valid'] += [voxel_rel_diff.item()]
                    traces['losses']['valid'] += [loss.item()]
                    traces['kl_divs']['valid'] += [kl_div.item()]
                    traces['jaccard']['valid'] += [jaccard.item()]
                    traces['dice']['valid'] += [dice.item()]
                    traces['recon']['valid'] += [loss_recon.item()]
                    traces['abs_error']['valid'] += [float(torch.mean(torch.abs_(reconstruct - images.to(device))).item())]
                results['losses']["valid"] += [np.mean(traces['losses']['valid'])]
                results['kl_divs']["valid"] += [np.mean(traces['kl_divs']['valid'])]
                results['recon']["valid"] += [np.mean(traces['recon']['valid'])]
                results['abs_error']["valid"] += [np.mean(traces['abs_error']['valid'])]
                results['acc']["valid"] += [np.mean(traces['acc']['valid'])]
                results['jaccard']["valid"] += [np.mean(traces['jaccard']['valid'])]
                results['dice']["valid"] += [np.mean(traces['dice']['valid'])]
                results["voxel_rel_diff"]["valid"] += [np.mean(traces['voxel_rel_diff']['valid'])]

                logger.add_scalar('valid/loss', results['losses']['valid'][-1], epoch + 1)
                logger.add_scalar('valid/kld', results['kl_divs']['valid'][-1], epoch + 1)
                logger.add_scalar('valid/recon', results['recon']['valid'][-1], epoch + 1)
                logger.add_scalar('valid/acc', results['acc']['valid'][-1], epoch + 1)
                logger.add_scalar('valid/jaccard', results['jaccard']['valid'][-1], epoch + 1)
                logger.add_scalar('valid/dice', results['dice']['valid'][-1], epoch + 1)
                logger.add_scalar('valid/voxel_rel_diff', results['voxel_rel_diff']['valid'][-1], epoch + 1)
                if self.scheduler == "ReduceLROnPlateau":
                    if epoch - epoch_offset > 5:
                        lr_schedule.step(results['losses']["valid"][-1])
                if (results['losses'][self.mode][-1] < best_loss or best_loss == -1) and not np.isnan(results['losses'][self.mode][-1]):
                    if self.verbose > 1:
                        print('BEST EPOCH!', results['losses'][self.mode][-1])
                    early_stop_counter = 0
                    best_loss = results['losses'][self.mode][-1]
                    is_best_epoch = True
                    best_epoch = epoch
                else:
                    early_stop_counter += 1

                if epoch % self.epochs_per_checkpoint == 0:
                    if self.save and is_best_epoch:
                        if f'views/{self.model_name}' not in os.listdir():
                            os.makedirs(f'views/{self.model_name}', exist_ok=True)
                        target = nib.Nifti1Image(targets.detach().cpu().numpy()[0], np.eye(4))
                        img = nib.Nifti1Image(images.detach().cpu().numpy()[0], np.eye(4))
                        recon = nib.Nifti1Image(reconstruct.detach().cpu().numpy()[0], np.eye(4))
                        bin_recon = nib.Nifti1Image(bin_reconstruct.detach().cpu().numpy()[0], np.eye(4))
                        target.to_filename(filename=f'views/{self.model_name}/target_valid.nii.gz')
                        img.to_filename(filename=f'views/{self.model_name}/image_valid.nii.gz')
                        recon.to_filename(filename=f'views/{self.model_name}/reconstruct_valid.nii.gz')
                        bin_recon.to_filename(filename=f'views/{self.model_name}/bin_reconstruct_valid.nii.gz')
                        middle = 3
                        logger.add_image(tag='valid/target', img_tensor=targets.detach().cpu()[0], global_step=epoch)
                        logger.add_image(tag='valid/reconstruction', img_tensor=reconstruct.detach().cpu()[0], global_step=epoch)
                        logger.add_image(tag='valid/bin_reconstruction', img_tensor=bin_reconstruct.detach().cpu()[0], global_step=epoch)
                        logger.add_image(tag='valid/input', img_tensor=images.detach().cpu()[0], global_step=epoch)
                    if is_best_epoch and self.save_checkpoints:
                        if self.verbose > 1:
                            print('Saving model...')
                        save_checkpoint(model=model,
                                        optimizer=optimizer,
                                        maxpool=maxpool,
                                        padding=self.padding,
                                        padding_deconv=self.padding_deconv,
                                        learning_rate=learning_rate,
                                        epoch=epoch,
                                        checkpoint_path=output_directory,
                                        z_dim=z_dim,
                                        gated=self.gated,
                                        batchnorm=self.batchnorm,
                                        losses=results['losses'],
                                        kl_divs=results['kl_divs'],
                                        losses_recon=results['recon'],
                                        in_channels=self.in_channels,
                                        out_channels=self.out_channels,
                                        kernel_sizes=self.kernel_sizes,
                                        kernel_sizes_deconv=self.kernel_sizes_deconv,
                                        strides=self.strides,
                                        strides_deconv=self.strides_deconv,
                                        dilatations=self.dilatations,
                                        dilatations_deconv=self.dilatations_deconv,
                                        best_loss=best_loss,
                                        save=self.save,
                                        name=self.model_name,
                                        n_flows=None,
                                        flow_type=self.flow_type,
                                        n_res=5,
                                        resblocks=resblocks,
                                        h_last=z_dim,
                                        n_elements=None,
                                        n_kernels=self.kernel_sizes
                                        )
                if epoch % self.epochs_per_print == 0:
                    if self.verbose > 0:
                        print("Epoch: {}:\t"
                              "Valid Loss: {:.5f} , "
                              "kld: {:.3f} , "
                              "recon: {:.3f}, "
                              "acc: {:.2f}, "
                              .format(epoch,
                                      results['losses']["valid"][-1],
                                      results['kl_divs']["valid"][-1],
                                      results['losses']["valid"][-1],
                                      results['acc']["valid"][-1],
                                      ),
                              f"Jaccard: {np.round(results['jaccard']['valid'][-1], 4)}, "
                              f"Dice: {np.round(results['dice']['valid'][-1], 4)}, "
                              f"Voxel Rel Diff: {np.round(results['voxel_rel_diff']['valid'][-1], 4)}"

                              )

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

        # TODO add to HPARAMS best metrics
        n_res = params['n_res']
        z_dim = params['z_dim']
        learning_rate = params['learning_rate'].__format__('e')
        n_flows = params['n_flows']
        weight_decay = params['weight_decay'].__format__('e')
        l1 = params['l1'].__format__('e')
        sensitivity, specificity = self.compute_confusion_matrix(y_test=targets, y_classes=bin_reconstruct)

        logger.add_hparams(
            {
                'n_res': n_res,
                'z_dim': z_dim,
                'lr': learning_rate,
                'n_flows': n_flows,
                'wd': weight_decay,
                # 'l1': l1,
            },
            {
                'hparam/train_accuracy': results['acc']['train'][best_epoch-1],
                'hparam/valid_accuracy': results['acc']['valid'][best_epoch-1],
                'hparam/train_loss': results['losses']['train'][best_epoch-1],
                'hparam/valid_loss': results['losses']['valid'][best_epoch-1],
                'hparam/train_kld': results['kl_divs']['train'][best_epoch-1],
                'hparam/valid_kld': results['kl_divs']['valid'][best_epoch-1],
                'hparam/train_recon': results['recon']['train'][best_epoch-1],
                'hparam/valid_recon': results['recon']['valid'][best_epoch-1],
                'hparam/jaccard': jaccard,
                'hparam/dice': dice,
                'hparam/sensitivity': sensitivity,
                'hparam/specificity': specificity,
            }
        )
        return np.mean(best_losses)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str, default="data\\canis_intensities.csv",
                        help="Path to intensities csv file")
    parser.add_argument("--labels_path", type=str, default="data\\canis_labels.csv",
                        help="Path to labels csv file")
    parser.add_argument("--verbose", type=str, default=1)
    args = parser.parse_args()
    torch.manual_seed(42)

    random.seed(42)

    size = 256
    z_dim = 500
    in_channels = [1, 8, 8]
    out_channels = [8, 8, 8]
    kernel_sizes = [3, 3, 3]
    kernel_sizes_deconv = [3, 3, 3]
    strides = [1, 1, 1]
    strides_deconv = [1, 1, 1]
    dilatations = [1, 1, 1]
    dilatations_Deconv = [1, 1, 1]
    paddings = [1, 1, 0]
    paddings_deconv = [0, 1, 1]
    dilatations_deconv = [1, 1, 1]
    in_channels2 = [1, 64, 128]
    out_channels2 = [64, 128, 256]
    paddings2 = [1, 1, 1]
    paddings_deconv2 = [2, 1, 1]
    n_flows = 10
    bs = 3
    maxpool = 2
    maxpool2 = 2
    flow_type = 'vanilla'
    epochs_per_checkpoint = 1
    has_dense = True
    batchnorm = True
    gated = False
    resblocks = True
    checkpoint_path = "checkpoints"
    images_path = 'D:\\workbench\\projects\\AutoTKV_MouseMRI-master\\AllTrainingImages\\images\\'

    n_epochs = 10000
    save = True
    training = Train(in_channels=in_channels,
                     in_channels2=in_channels2,
                     out_channels=out_channels,
                     out_channels2=out_channels2,
                     kernel_sizes=kernel_sizes,
                     kernel_sizes_deconv=kernel_sizes_deconv,
                     strides=strides,
                     strides_deconv=strides_deconv,
                     dilatations=dilatations,
                     dilatations_deconv=dilatations_deconv,
                     images_path=args.images_path,
                     targets_path=args.labels_path,
                     padding=paddings,
                     padding2=paddings2,
                     padding_deconv=paddings_deconv,
                     padding_deconv2=paddings_deconv2,
                     batch_size=bs,
                     epochs=n_epochs,
                     checkpoint_path=checkpoint_path,
                     epochs_per_checkpoint=epochs_per_checkpoint,
                     gated=gated,
                     resblocks=resblocks,
                     fp16_run=False,
                     batchnorm=batchnorm,
                     flow_type=flow_type,
                     save=save,
                     maxpool=maxpool,
                     maxpool2=maxpool2,
                     plot_perform=False,
                     activation=torch.nn.ReLU,
                     mean=0.1,
                     std=0.1,
                     init_func=torch.nn.init.xavier_uniform_,
                     mode='valid',
                     load=False,
                     cv=5,
                     size=size
                     )

    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "z_dim", "type": "range", "bounds": [256, 512]},
            # {"name": "l1", "type": "range", "bounds": [1e-8, 1e-1], "log_scale": True},
            {"name": "weight_decay", "type": "range", "bounds": [1e-8, 1e-4], "log_scale": True},
            {"name": "learning_rate", "type": "range", "bounds": [1e-5, 1e-4], "log_scale": True},
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
