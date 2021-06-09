import os
import torch
import itertools
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
from fmri.models.supervised.resnetcnn3d import ConvResnet3D
from fmri.models.unsupervised.VAE_3DCNN import Autoencoder3DCNN
from fmri.models.unsupervised.SylvesterVAE3DCNN import SylvesterVAE
import random
import pydicom
import datetime
import pandas as pd
from fmri.utils.transform_3d import Normalize, Rotation3D, ColorJitter3D, Flip90, Flip180, Flip270, XFlip, YFlip, \
    ZFlip, RandomAffine3D

from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, Resize, ToPILImage, \
    ToTensor
import torchvision
random.seed(42)


def resize_data(data, new_size=(160, 160, 160)):
    initial_size_x = data.shape[0]
    initial_size_y = data.shape[1]
    initial_size_z = data.shape[2]

    new_size_x = new_size[0]
    new_size_y = new_size[1]
    new_size_z = new_size[2]

    delta_x = initial_size_x / new_size_x
    delta_y = initial_size_y / new_size_y
    delta_z = initial_size_z / new_size_z

    new_data = np.zeros((new_size_x, new_size_y, new_size_z))

    for x, y, z in itertools.product(range(new_size_x),
                                     range(new_size_y),
                                     range(new_size_z)):
        new_data[x][y][z] = data[int(x * delta_x)][int(y * delta_y)][int(z * delta_z)]

    return new_data


def resize_data2d(data, new_size=(160, 160)):
    initial_size_x = data.shape[0]
    initial_size_y = data.shape[1]

    new_size_x = new_size[0]
    new_size_y = new_size[1]

    delta_x = initial_size_x / new_size_x
    delta_y = initial_size_y / new_size_y

    new_data = np.zeros((new_size_x, new_size_y))

    for x, y in itertools.product(range(new_size_x),
                                  range(new_size_y)):
        new_data[x][y] = data[int(x * delta_x)][int(y * delta_y)]

    return new_data


class MRIDataset3D(Dataset):
    def __init__(self, path, targets_path, transform=True, normalize=None, size=16, resize=True, device='cuda'):
        self.path = path
        self.targets_path = targets_path
        self.device = device
        self.size = size
        self.samples = os.listdir(path)
        self.targets = os.listdir(targets_path)
        # random.shuffle(self.samples)
        self.transform = transform
        self.normalize = normalize
        self.resize = resize
        (self.xflip, self.yflip, self.zflip, self.flip90, self.flip180, self.flip270, self.rot0, self.rot1,
         self.rot2) = [0 for _ in range(9)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        self.samples.sort()
        self.targets.sort()
        if self.transform:
            (self.xflip, self.yflip, self.zflip, self.flip90, self.flip180, self.flip270, self.rot0, self.rot1,
             self.rot2) = [random.randint(0, 1) for _ in range(9)]
        x = self.samples[idx]
        x = nib.load(self.path + x).dataobj
        x = np.array(x)

        target = self.targets[idx]
        target = nib.load(self.targets_path + target).dataobj
        target = np.array(target)

        # TODO replace by max of all targets?
        voxel_count = np.sum(target / np.max(target))

        if self.resize:
            x = resize_data(x, (self.size, self.size, 16))
            target = resize_data(target, (self.size, self.size, 16))
        x = torch.Tensor(x)  # .to(self.device)
        target = torch.Tensor(target)  # .to(self.device)
        # x.requires_grad = False
        if self.xflip:
            x = XFlip()(x)
            target = XFlip()(target)
        if self.yflip:
            x = YFlip()(x)
            target = YFlip()(target)
        if self.zflip:
            x = ZFlip()(x)
            target = ZFlip()(target)

        if self.flip90:
            x = Flip90()(x)
            target = Flip90()(target)

        if self.flip180:
            x = Flip180()(x)
            target = Flip180()(target)

        if self.flip270:
            x = Flip270()(x)
            target = Flip270()(target)

        if self.rot0:
            x = Rotation3D(90, 0)(x)
            target = Rotation3D(90, 0)(target)

        if self.rot1:
            x = Rotation3D(90, 1)(x)
            target = Rotation3D(90, 1)(target)

        if self.rot2:
            x = Rotation3D(90, 2)(x)
            target = Rotation3D(90, 2)(target)

        if self.normalize:
            x = self.normalize(x)
            target = self.normalize(target)
            # target = target / target.max()
        if x.unsqueeze(0).shape != target.shape:
            pass
        new_voxel_count = torch.sum(target / torch.max(target))
        voxel_ratio = new_voxel_count / voxel_count
        return x.unsqueeze(0), target, voxel_count, voxel_ratio


class MRIDataset2D(Dataset):
    def __init__(self, path, targets_path, transform=True, normalize=None, size=16, resize=True, device='cuda',
                 test=False):
        self.path = path
        self.targets_path = targets_path
        self.device = device
        self.size = size
        self.samples = os.listdir(path)
        self.targets = os.listdir(targets_path)
        # random.shuffle(self.samples)
        self.transform = transform
        self.normalize = normalize
        self.resize = resize
        self.test = test
        (self.xflip, self.yflip, self.rot, self.rot1, self.rot2) = [False for _ in range(5)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # The names need to be sorted or at some point an incompatible pair is made
        self.samples.sort()
        self.targets.sort()
        if self.transform:
            (self.xflip, self.yflip, self.rot, self.rot1, self.rot2) = [True if x == 1 else False for x in
                                                                        [random.randint(0, 1) for _ in range(5)]]
        x = self.samples[idx]
        x = nib.load(self.path + x).dataobj

        # Sum on dim z
        target = self.targets[idx]
        target = nib.load(self.targets_path + target).dataobj

        target = torch.Tensor(np.array(target))
        x = torch.Tensor(np.array(x))
        ind = random.randint(0, x.shape[2] - 1)
        x = x[:, :, ind]
        target = target[:, :, ind]

        x = resize_data2d(x, (self.size, self.size))
        target = resize_data2d(target, (self.size, self.size))
        x = x.reshape([x.shape[0], x.shape[1]])
        x = ToPILImage()(np.array(x, dtype=int))
        target = ToPILImage()(np.array(target, dtype=int))

        if self.xflip:
            # xflip is random, so we need them both flipped together
            x = RandomHorizontalFlip(p=1)(x)
            target = RandomHorizontalFlip(p=1)(target)
        if self.yflip:
            x = RandomVerticalFlip(p=1)(x)
            target = RandomVerticalFlip(p=1)(target)

        if self.rot:
            x = torchvision.transforms.functional.rotate(x, 90)
            target = torchvision.transforms.functional.rotate(target, 90)
        if self.rot1:
            x = torchvision.transforms.functional.rotate(x, 180)
            target = torchvision.transforms.functional.rotate(target, 180)
        # if self.rot2:
        #     x = torchvision.transforms.functional.rotate(x, 270)
        #     target = torchvision.transforms.functional.rotate(target, 270)

        x = torch.Tensor(np.array(x))
        target = torch.Tensor(np.array(target))

        if self.normalize:
            x[torch.isnan(x)] = 0
            target[torch.isnan(target)] = 0
            x = x.float()
            target = target.float()
            if x.max() > 0:
                x /= x.max()
            if target.max() > 0:
                target /= target.max()
        # TODO replace by max of all targets?

        target[target > 0] = 1

        return x, target


def load_scan(path):
    """
    Loads scans from a folder and into a list.

    Parameters: path (Folder path)

    Returns: slices (List of slices)
    """
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(scans):
    """
    Converts raw images to Hounsfield Units (HU).

    Parameters: scans (Raw images)

    Returns: image (NumPy array)
    """

    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    # Since the scanning equipment is cylindrical in nature and image output is square,
    # we set the out-of-scan pixels to 0
    image[image == -2000] = 0

    # HU = m*P + b
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def load_checkpoint(params,
                    checkpoint_path,
                    model,
                    optimizer,
                    predict,
                    name,
                    model_name,
                    epoch,
                    timestamp
                    ):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    losses = {
        "train": [],
        "valid": [],
    }
    if checkpoint_path not in os.listdir() and not predict:
        os.makedirs(checkpoint_path, exist_ok=True)

    names = os.listdir(f"{checkpoint_path}/{name}")
    names.sort()
    if len(names) == 0 and not predict:
        print("Creating checkpoint...")
        last_name = timestamp
        save_checkpoint(model=model,
                        optimizer=optimizer,
                        params=params,
                        checkpoint_path=checkpoint_path,
                        losses=losses,
                        name=name,
                        model_name=model_name,
                        epoch=epoch,
                        timestamp=last_name
                        )
    else:
        print('Checkpoint exists.')
        last_name = names[-1]
    checkpoint_dict = torch.load(f"{checkpoint_path}/{name}/{last_name}", map_location=device)
    epoch = checkpoint_dict['epoch']
    best_loss = checkpoint_dict['best_loss']
    # optimizer.load_state_dict(checkpoint_dict['optimizer'].state_dict().to(device))
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    model = model.to(device)
    losses = checkpoint_dict['losses']
    print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    return model, optimizer, epoch, losses, best_loss


def save_checkpoint(
        model,
        optimizer,
        params,
        epoch,
        checkpoint_path,
        losses,
        model_name,
        name,
        timestamp,
        best_loss=-1,
        n_classes=None
):
    model_for_saving = None
    if name == 'classifier':
        model_for_saving = model_name(params)
    model_type = name.split("_")[0]
    if model_type == 'vae':
        if params['flow_type'] != 'o-sylvester':
            model_for_saving = model_name(params)
        else:
            model_for_saving = SylvesterVAE(params)
    elif model_type == "classif":
        model_for_saving = ConvResnet3D(params, n_classes)
    elif model_type == 'vqvae':
        model_for_saving = model_name(
            in_channel=1,
            channel=512,
            n_res_block=params['n_res'],
            n_res_channel=64,
            embed_dim=params['z_dim'],
            n_embed=1024,
        )
    else:
        exit(f'{model_type} cannot be saved for now. Sorry.')
    model_for_saving.load_state_dict(model.state_dict())
    os.makedirs(checkpoint_path + '/' + name, exist_ok=True)
    torch.save({
        'model': model_for_saving,
        'optimizer': optimizer,
        'losses': losses,
        'best_loss': best_loss,
        'epoch': epoch,
        'learning_rate': params['lr']
    }, f"{checkpoint_path}/{name}/{timestamp}.model")
