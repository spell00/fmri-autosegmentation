import os
import json
import torch
import random
import datetime
import argparse
import numpy as np
import nibabel as nib
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from ax.service.managed_loop import optimize

from fmri.utils.utils import validation_spliter
from fmri.models.unsupervised.VQVAE2_2D import VQVAE
from fmri.utils.plot_performance import plot_performance
from fmri.utils.CycleAnnealScheduler import CycleScheduler
from fmri.utils.dataset import load_checkpoint, save_checkpoint, MRIDataset2D

torch.manual_seed(42)
random.seed(42)


class DiceLoss(nn.Module):
    # source: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, y_true, y_preds, smooth=1):
        inputs = y_preds.view(-1)
        targets = y_true.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return -dice


class JaccardLoss(nn.Module):
    # source: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
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

        return -jaccard


class MCCLoss(nn.Module):
    def __init__(self):
        super(MCCLoss, self).__init__()

    def forward(self, y_true, y_preds):
        exit('not implemented')
        pass


dice_loss = DiceLoss()
jaccard_loss = JaccardLoss()


def compute_confusion_matrix(y_test, y_classes):
    ys = torch.round(y_test.reshape([y_test.shape[0], -1]))
    y_classes = y_classes.reshape([y_classes.shape[0], -1])
    tp = np.sum([[1 if true == 1 and pred == 1 else 0 for true, pred in zip(t, c)] for t, c in zip(ys, y_classes)])
    tn = np.sum([[1 if true == 0 and pred == 0 else 0 for true, pred in zip(t, c)] for t, c in zip(ys, y_classes)])
    fp = np.sum([[1 if true == 0 and pred == 1 else 0 for true, pred in zip(t, c)] for t, c in zip(ys, y_classes)])
    fn = np.sum([[1 if true == 1 and pred == 0 else 0 for true, pred in zip(t, c)] for t, c in zip(ys, y_classes)])

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    return sensitivity, specificity


def get_optimizer(model, params):
    if params['optimizer_type'] == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=params['lr'],
                                     weight_decay=params['wd']
                                     )
    elif params['optimizer_type'] == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=params['lr'],
                                    weight_decay=params['wd'],
                                    momentum=0.9)
    elif params['optimizer_type'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(params=model.parameters(),
                                        lr=params['lr'],
                                        weight_decay=params['wd'],
                                        momentum=0.9)
    else:
        exit('error: no such optimizer type available')
    return optimizer


def print_results(results, mode):
    print(
        f"Epoch: {len(results['losses'][mode])}\t"
        f"{mode} Loss: {np.round(results['losses'][mode][-1], 4)} , "
        f"kld: {np.round(results['kl_divs'][mode][-1], 4)} , "
        f"recon: {np.round(results['recon'][mode][-1], 4)}, "
        f"acc: {np.round(results['acc'][mode][-1], 4)} "
        f"Jaccard: {np.round(results['jaccard'][mode][-1], 4)}, "
        f"Dice: {np.round(results['dice'][mode][-1], 4)} "
    )


class Train:
    def __init__(self,
                 params,
                 images_path,
                 targets_path,
                 loss,
                 checkpoint_path,
                 fp16_run=False,
                 epochs_per_checkpoint=1,
                 epochs_per_print=1,
                 verbose=2,
                 # val_share=0.1,
                 # mode='valid',
                 early_stop=100,
                 load=True,
                 save_checkpoints=False,
                 save=True,
                 ):
        super().__init__()
        self.fp16_run = fp16_run
        self.checkpoint_path = checkpoint_path
        self.epochs_per_checkpoint = epochs_per_checkpoint
        self.epochs_per_print = epochs_per_print
        self.save = save
        if loss == 'bce':
            self.criterion = nn.BCELoss()
        elif loss == 'dice':
            self.criterion = dice_loss
        elif loss == 'jaccard':
            self.criterion = jaccard_loss
        else:
            exit(f'{loss} is not implemented. Valid choices are: [bce, dice, jaccard]')

        self.save_checkpoints = save_checkpoints
        self.load = load
        self.loss_type = loss
        self.verbose = verbose
        self.images_path = images_path
        self.targets_path = targets_path
        # self.val_share = val_share
        # self.mode = mode
        self.early_stop = early_stop
        self.model_name = ''
        self.params = params

    def print_params(self):
        print(
            f"Parameters: \n\t",
            f'z_dim: {self.params["z_dim"]}\n\t',
            f'n_res: {self.params["n_res"]}\n\t',
            f'learning_rate: {self.params["lr"].__format__("e")} \n\t',
            f'weight_decay: {self.params["wd"].__format__("e")} \n\t',
            f'l1: {self.params["l1"]}\n\t',
            f'optimizer_type: {self.params["optimizer_type"]}\n\t',
            f'in_channels: {"-".join([str(item) for item in self.params["in_channels"]])}\n\t',
            f'out_channels: {"-".join([str(item) for item in self.params["out_channels"]])}\n\t',
            f'kernel_sizes: {"-".join([str(item) for item in self.params["kernel_sizes"]])}\n\t',
            f'kernel_sizes_deconv: {"-".join([str(item) for item in self.params["kernel_sizes_deconv"]])}\n\t',
            f'paddings: {"-".join([str(item) for item in self.params["paddings"]])}\n\t',
            f'paddings_deconv: {"-".join([str(item) for item in self.params["paddings_deconv"]])}\n\t',
            f'dilatations: {"-".join([str(item) for item in self.params["dilatations"]])}\n\t',
            f'dilatations_deconv: {"-".join([str(item) for item in self.params["dilatations_deconv"]])}\n\t'
        )

    def train(self, params_dict):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        best_losses = []

        # n_res = params['n_res']
        self.params['z_dim'] = params_dict['z_dim']
        self.params['lr'] = params_dict['learning_rate']
        self.params['n_res'] = params_dict['n_res']
        self.params['wd'] = params_dict['weight_decay']
        timestamp = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
        self.model_name = '/'.join([
            f'vqvae_2d_cnn',
            f'{self.loss_type}',
            f'{self.params["optimizer_type"]}',
            f'n_res{self.params["n_res"]}',
            f'z_dim{self.params["z_dim"]}',
            f'wd{self.params["wd"]}',
            f'l1{self.params["l1"]}',
        ])
        if self.verbose > 1:
            self.print_params()
        model = VQVAE(
            in_channel=1,
            channel=512,
            n_res_block=self.params['n_res'],
            n_res_channel=64,
            embed_dim=self.params['z_dim'],
            n_embed=1024,
        )
        model.random_init()
        optimizer = get_optimizer(model, self.params)

        # Load checkpoint if one exists
        epoch = 0
        best_loss = -1

        if self.checkpoint_path is not None and self.load:
            try:
                model, _, \
            epoch, losses, \
            best_loss = load_checkpoint(checkpoint_path=self.checkpoint_path,
                                        model=model,
                                        params=self.params,
                                        epoch=epoch,
                                        predict=False,
                                        optimizer=optimizer,
                                        name=self.model_name,
                                        model_name=VQVAE,
                                        timestamp=timestamp
                                        )
            except IOError:
                print('No checkpoint found. Creating a new model.')
        model = model.to(device)

        all_set = MRIDataset2D(self.images_path, self.targets_path, size=self.params['size'], normalize=True,
                               transform=True)
        spliter = validation_spliter(all_set, cv=self.params['cv'])

        epoch_offset = max(1, epoch)

        # for cv in range(self.cross_validation):
        for cv in range(1):
            model.random_init()
            best_loss = -1
            valid_set, train_set = spliter.__next__()

            train_loader = DataLoader(train_set,
                                      num_workers=0,
                                      shuffle=True,
                                      batch_size=self.params['bs'],
                                      pin_memory=False,
                                      drop_last=True)
            valid_loader = DataLoader(valid_set,
                                      num_workers=0,
                                      shuffle=True,
                                      batch_size=1,
                                      pin_memory=False,
                                      drop_last=True)

            os.makedirs('logs/' + self.model_name, exist_ok=True)
            logger = SummaryWriter('logs/' + self.model_name)

            lr_scheduler = self.get_scheduler(optimizer, len(train_loader))

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
            }

            early_stop_counter = 0
            print("\n\n\nCV:", cv, "/", self.params['cv'], "\nTrain samples:", len(train_set),
                  "\nValid samples:", len(valid_set), "\n\n\n")

            is_best_epoch = False
            best_epoch = 0
            for epoch in range(epoch_offset, self.params['n_epochs']):
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
                    images, targets = batch
                    images = images.to(device)
                    targets = targets.to(device)
                    reconstruct, kl = model(images.unsqueeze(1))
                    bin_reconstruct = torch.round(reconstruct)
                    jaccard = jaccard_loss(targets, bin_reconstruct)
                    dice = dice_loss(targets, bin_reconstruct)

                    loss_recon = self.criterion(
                        reconstruct.squeeze(1),
                        targets
                    ).sum() / self.params['bs']
                    kl_div = torch.mean(kl)
                    loss = loss_recon + kl_div # + 0.1 * dice
                    if self.params['l1'] > 0:
                        l1 = self.get_l1()
                        loss += l1
                    loss.backward()

                    total = len(targets[0].view(-1))
                    acc = np.mean([torch.sum(x == y).item() for x, y in zip(bin_reconstruct, targets)]) / total

                    traces['acc']['train'] += [acc]
                    traces['losses']['train'] += [loss.item()]
                    traces['kl_divs']['train'] += [kl_div.item()]
                    traces['recon']['train'] += [loss_recon.item()]
                    traces['dice']['train'] += [-dice.item()]
                    traces['jaccard']['train'] += [-jaccard.item()]
                    traces['abs_error']['train'] += [
                        float(torch.mean(torch.abs_(
                            reconstruct - images.to(device)
                        )).item())
                    ]

                    optimizer.step()
                    if self.params['scheduler'] == "CycleScheduler":
                        lr_scheduler.step()
                        # optimizer = lr_scheduler.optimizer
                    del kl, loss_recon, kl_div, loss, dice

                target = nib.Nifti1Image(targets.detach().cpu().numpy()[0], np.eye(4))
                img = nib.Nifti1Image(images.detach().cpu().numpy()[0], np.eye(4))
                recon = nib.Nifti1Image(reconstruct.detach().cpu().numpy()[0][0], np.eye(4))
                bin_recon = nib.Nifti1Image(bin_reconstruct.detach().cpu().numpy()[0][0], np.eye(4))
                if self.save and is_best_epoch:
                    if f'views/{self.model_name}' not in os.listdir():
                        os.makedirs(f'views/{self.model_name}', exist_ok=True)
                    target.to_filename(filename=f'views/{self.model_name}/target_train_.nii.gz')
                    img.to_filename(filename=f'views/{self.model_name}/image_train.nii.gz')
                    recon.to_filename(filename=f'views/{self.model_name}/reconstruct_train.nii.gz')
                    bin_recon.to_filename(filename=f'views/{self.model_name}/reconstruct_train_bin.nii.gz')
                    middle = 3
                    logger.add_image(tag='train/target', img_tensor=targets.detach().cpu()[0].unsqueeze(0),
                                     global_step=epoch)
                    logger.add_image(tag='train/reconstruction', img_tensor=reconstruct.detach().cpu()[0],
                                     global_step=epoch)
                    logger.add_image(tag='train/binary_reconstruction', img_tensor=bin_reconstruct.detach().cpu()[0],
                                     global_step=epoch)
                    logger.add_image(tag='train/input', img_tensor=images.detach().cpu()[0].unsqueeze(0),
                                     global_step=epoch)
                results["losses"]["train"] += [np.mean(traces['losses']['train'])]
                results["kl_divs"]["train"] += [np.mean(traces['kl_divs']['train'])]
                results["recon"]["train"] += [np.mean(traces['recon']['train'])]
                results["abs_error"]["train"] += [np.mean(traces['abs_error']['train'])]
                results["acc"]["train"] += [np.mean(traces['acc']['train'])]
                results["jaccard"]["train"] += [-np.mean(traces['jaccard']['train'])]
                results["dice"]["train"] += [-np.mean(traces['dice']['train'])]

                logger.add_scalar('train/loss', results["losses"]["train"][-1], epoch + 1)
                logger.add_scalar('train/kld', results["kl_divs"]["train"][-1], epoch + 1)
                logger.add_scalar('train/recon', results["recon"]["train"][-1], epoch + 1)
                logger.add_scalar('train/accuracy', results["acc"]["train"][-1], epoch + 1)
                logger.add_scalar('train/jaccard', results["jaccard"]["train"][-1], epoch + 1)
                logger.add_scalar('train/dice', results["dice"]["train"][-1], epoch + 1)

                if epoch % self.epochs_per_print == 0:
                    if self.verbose > 1:
                        print_results(results, mode='train')

                model.eval()
                for i, batch in enumerate(valid_loader):
                    images, targets = batch
                    images = images.to(device)
                    targets = targets.to(device)
                    reconstruct, kl = model(images.unsqueeze(1))
                    bin_reconstruct = torch.round(reconstruct)
                    loss_recon = self.criterion(
                        reconstruct.squeeze(1),
                        targets
                    ).sum()
                    kl_div = torch.mean(kl)
                    dice = dice_loss(y_true=targets, y_preds=bin_reconstruct)
                    loss = loss_recon + kl_div  # + 0.1 * dice
                    jaccard = jaccard_loss(
                        y_true=targets.reshape(targets.shape[0], -1),
                        y_preds=bin_reconstruct.reshape(targets.shape[0], -1)
                    )

                    total = len(targets[0].view(-1))
                    acc = np.mean([torch.sum(x == y).item() for x, y in zip(bin_reconstruct, targets)]) / total
                    traces['acc']['valid'] += [acc]
                    traces['losses']['valid'] += [loss.item()]
                    traces['kl_divs']['valid'] += [kl_div.item()]
                    traces['jaccard']['valid'] += [-jaccard.item()]
                    traces['dice']['valid'] += [-dice.item()]
                    traces['recon']['valid'] += [loss_recon.item()]
                    traces['abs_error']['valid'] += [
                        float(torch.mean(torch.abs_(reconstruct - images.to(device))).item())]
                results['losses']["valid"] += [np.mean(traces['losses']['valid'])]
                results['kl_divs']["valid"] += [np.mean(traces['kl_divs']['valid'])]
                results['recon']["valid"] += [np.mean(traces['recon']['valid'])]
                results['abs_error']["valid"] += [np.mean(traces['abs_error']['valid'])]
                results['acc']["valid"] += [np.mean(traces['acc']['valid'])]
                results['jaccard']["valid"] += [np.mean(traces['jaccard']['valid'])]
                results['dice']["valid"] += [-np.mean(traces['dice']['valid'])]

                logger.add_scalar('valid/loss', results['losses']['valid'][-1], epoch + 1)
                logger.add_scalar('valid/kld', results['kl_divs']['valid'][-1], epoch + 1)
                logger.add_scalar('valid/recon', results['recon']['valid'][-1], epoch + 1)
                logger.add_scalar('valid/acc', results['acc']['valid'][-1], epoch + 1)
                logger.add_scalar('valid/jaccard', results['jaccard']['valid'][-1], epoch + 1)
                logger.add_scalar('valid/dice', -results['dice']['valid'][-1], epoch + 1)
                if self.params['scheduler'] == "ReduceLROnPlateau":
                    if epoch - epoch_offset > 5:
                        lr_scheduler.step(results['losses']["valid"][-1])
                if (results['losses']['valid'][-1] < best_loss or best_loss == -1) and not np.isnan(
                        results['losses']['valid'][-1]):
                    if self.verbose > 1:
                        print('BEST EPOCH!', results['losses']['valid'][-1])
                    early_stop_counter = 0
                    best_loss = results['losses']['valid'][-1]
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
                        recon = nib.Nifti1Image(reconstruct.detach().cpu().numpy()[0][0], np.eye(4))
                        bin_recon = nib.Nifti1Image(bin_reconstruct.detach().cpu().numpy()[0][0], np.eye(4))
                        target.to_filename(filename=f'views/{self.model_name}/target_valid.nii.gz')
                        img.to_filename(filename=f'views/{self.model_name}/image_valid.nii.gz')
                        recon.to_filename(filename=f'views/{self.model_name}/reconstruct_valid.nii.gz')
                        bin_recon.to_filename(filename=f'views/{self.model_name}/bin_reconstruct_valid.nii.gz')

                        logger.add_image(tag='valid/target', img_tensor=targets.detach().cpu()[0].unsqueeze(0),
                                         global_step=epoch)
                        logger.add_image(tag='valid/reconstruction', img_tensor=reconstruct.detach().cpu()[0],
                                         global_step=epoch)
                        logger.add_image(tag='valid/bin_reconstruction', img_tensor=bin_reconstruct.detach().cpu()[0],
                                         global_step=epoch)
                        logger.add_image(tag='valid/input', img_tensor=images.detach().cpu()[0].unsqueeze(0),
                                         global_step=epoch)
                    if is_best_epoch and self.save_checkpoints:
                        if self.verbose > 1:
                            print('Saving model...')
                        losses_to_save = {
                            'losses': results['losses'],
                            'kl_divs': results['kl_divs'],
                            'recon': results['recon'],
                        }

                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            checkpoint_path=self.checkpoint_path,
                            losses=losses_to_save,
                            best_loss=best_loss,
                            name=self.model_name,
                            params=self.params,
                            model_name=VQVAE,
                            timestamp=timestamp
                        )
                if epoch % self.epochs_per_print == 0:
                    if self.verbose > 0:
                        print_results(results, mode='valid')

                    if self.verbose > 1:
                        print("Current LR:", optimizer.param_groups[0]['lr'])
                    if 'momentum' in optimizer.param_groups[0].keys():
                        print("Current Momentum:", optimizer.param_groups[0]['momentum'])
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
        sensitivity, specificity = compute_confusion_matrix(y_test=targets, y_classes=bin_reconstruct)

        logger.add_hparams(
            {
                'n_res': n_res,
                'z_dim': z_dim,
                'lr': learning_rate,
                'n_flows': n_flows,
                'wd': weight_decay,
                'l1': l1,
            },
            {
                'hparam/train_accuracy': results['acc']['train'][best_epoch - 1],
                'hparam/valid_accuracy': results['acc']['valid'][best_epoch - 1],
                'hparam/train_loss': results['losses']['train'][best_epoch - 1],
                'hparam/valid_loss': results['losses']['valid'][best_epoch - 1],
                'hparam/train_kld': results['kl_divs']['train'][best_epoch - 1],
                'hparam/valid_kld': results['kl_divs']['valid'][best_epoch - 1],
                'hparam/train_recon': results['recon']['train'][best_epoch - 1],
                'hparam/valid_recon': results['recon']['valid'][best_epoch - 1],
                'hparam/jaccard': -jaccard,
                'hparam/dice': -dice,
                'hparam/sensitivity': sensitivity,
                'hparam/specificity': specificity,
            }
        )
        return np.mean(best_losses)

    def get_scheduler(self, optimizer, n_samples=None):
        if self.params['scheduler'] == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                      factor=0.1,
                                                                      cooldown=0,
                                                                      patience=50,
                                                                      verbose=True,
                                                                      min_lr=1e-15)
        elif self.params['scheduler'] == 'CycleScheduler':
            assert n_samples is not None
            lr_scheduler = CycleScheduler(optimizer,
                                          self.params['lr'],
                                          n_iter=self.params['n_epochs'] * n_samples)
        else:
            lr_scheduler = None
        return lr_scheduler

    def get_l1(self):
        l1_reg = torch.tensor(0.)
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_reg += torch.norm(param, 1)

        return self.params['l1'] * l1_reg


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str, help="Path to intensities csv file")
    parser.add_argument("--labels_path", type=str, help="Path to labels csv file")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Checkpoint path")
    parser.add_argument("--loss", type=str, default="bce", help="Path to labels csv file")
    parser.add_argument("--verbose", type=str, default=1)
    args = parser.parse_args()

    params = {
        "cv": 5,
        "bn": True,
        "bs": 3,
        "n_epochs": 10000,
        "dilatations": [1, 1, 1],
        "dilatations_deconv": [1, 1, 1],
        "in_channels": [1, 8, 8],
        "out_channels": [8, 8, 8],
        "kernel_sizes": [3, 3, 3],
        "kernel_sizes_deconv": [3, 3, 3],
        "strides": [1, 1, 1],
        "strides_deconv": [1, 1, 1],
        "size": 256,
        "paddings": [1, 1, 0],
        "paddings_deconv": [0, 1, 1],
        'activation': torch.nn.ReLU,
        'n_embed': 2048,
        'epochs_per_checkpoint': 1,
        'max_pool': 3,
        'optimizer_type': 'adam',
        'scheduler': "ReduceLROnPlateau",
        "res_blocks": True,
        "has_dense": True,
        "z_dim": None,
        "lr": None,
        "wd": None,
        "n_res": None,
        "l1": 0

    }

    training = Train(
        params,
        images_path=args.images_path,
        targets_path=args.labels_path,
        loss=args.loss,
        checkpoint_path=args.checkpoint_path,
        save=True,
        load=True,
    )

    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "z_dim", "type": "range", "bounds": [256, 1024]},
            {"name": "n_res", "type": "range", "bounds": [1, 10]},
            # {"name": "l1", "type": "range", "bounds": [1e-8, 1e-1], "log_scale": True},
            {"name": "weight_decay", "type": "range", "bounds": [1e-12, 1e-8], "log_scale": True},
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
