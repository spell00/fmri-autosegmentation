import torch.nn.functional as F
from random import random, randint
from tqdm import tqdm
import torch
import numpy as np
import numbers
from PIL import Image
import torchvision


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this is_transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This is_transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        tensor = tensor - torch.min(tensor)
        return tensor / torch.max(tensor)

    def __repr__(self):
        return self.__class__.__name__


class Flip90(object):
    def __init__(self, rand=False, inplace=False):
        self.inplace = inplace
        self.rand = rand

    def __call__(self, tensor, p=0.5):
        if randint(0, 1) == 1 or not self.rand:
            return tensor.transpose(1, 2)
        else:
            return tensor


class XFlip(object):
    def __init__(self, rand=False, inplace=False):
        self.inplace = inplace
        self.rand = rand

    def __call__(self, tensor, p=0.5):
        if randint(0, 1) == 1 or not self.rand:
            return tensor.flip(0)
        else:
            return tensor


class YFlip(object):
    def __init__(self, rand=False, inplace=False):
        self.inplace = inplace
        self.rand = rand

    def __call__(self, tensor, p=0.5):
        if randint(0, 1) == 1 or not self.rand:
            return tensor.flip(1)
        else:
            return tensor


class ZFlip(object):
    def __init__(self, rand=False, inplace=False):
        self.inplace = inplace
        self.rand = rand

    def __call__(self, tensor, p=0.5):
        if randint(0, 1) == 1 or not self.rand:
            return tensor.flip(2)
        else:
            return tensor


class Flip180(object):
    def __init__(self, rand=False, inplace=False):
        self.inplace = inplace
        self.rand = rand

    def __call__(self, tensor, p=0.5):
        if randint(0, 1) == 1 or not self.rand:
            return tensor.transpose(1, 2).flip(0)
        else:
            return tensor


class Flip270(object):
    def __init__(self, rand=False, inplace=False):
        self.inplace = inplace
        self.rand = rand

    def __call__(self, tensor, p=0.5):
        if randint(0, 1) == 1 or not self.rand:
            return tensor.transpose(1, 2).flip(2)
        else:
            return tensor


# following classes adapted from https://raw.githubusercontent.com/perone/medicaltorch/master/medicaltorch/transforms.py

class Rotation3D(object):
    """Make a rotation of the volume's values.

    :param degrees: Maximum rotation's degrees.
    :param axis: Axis of the rotation.
    """

    def __init__(self, degrees, axis=0, labeled=True):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.labeled = labeled
        self.axis = axis

    def __call__(self, input_data):
        input_data = input_data.detach().cpu().numpy()
        input_rotated = np.zeros(input_data.shape, dtype=input_data.dtype)

        for x in range(input_data.shape[self.axis]):
            if self.axis == 0:
                pil = Image.fromarray(input_data[x, :, :], mode='F')
                input_rotated[x, :, :] = torchvision.transforms.functional.rotate(pil, self.degrees[0])
            if self.axis == 1:
                pil = Image.fromarray(input_data[:, x, :], mode='F')
                input_rotated[:, x, :] = torchvision.transforms.functional.rotate(pil, self.degrees[0])
            if self.axis == 2:
                pil = Image.fromarray(input_data[:, :, x], mode='F')
                input_rotated[:, :, x] = torchvision.transforms.functional.rotate(pil, self.degrees[0])
        return torch.Tensor(input_rotated)


class RandomAffine3D(object):
    def __init__(self, axis=0, translate=0.1, scale=0.1, shear=0.1):
        self.random_affine = torchvision.transforms.RandomAffine(translate, scale, shear)
        self.axis = axis

    def __call__(self, input_data):
        input_data = input_data.detach().cpu().numpy()
        input_affine = np.zeros(input_data.shape, dtype=input_data.dtype)

        for x in range(input_data.shape[0]):
            if self.axis == 0:
                input_affine[x, :, :] = self.random_affine(Image.fromarray(input_data[x, :, :], mode='L'))
            if self.axis == 1:
                input_affine[:, x, :] = self.random_affine(Image.fromarray(input_data[:, x, :], mode='L'))
            if self.axis == 2:
                input_affine[:, :, x] = self.random_affine(Image.fromarray(input_data[:, :, x], mode='L'))
        return torch.Tensor(input_affine) / 255


class ColorJitter3D(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.color_jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    @staticmethod
    def get_params(degrees):
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, input_data):
        input_data = input_data.detach().cpu().numpy()
        input_jittered = np.zeros(input_data.shape, dtype=input_data.dtype)

        for x in range(input_data.shape[0]):
            input_jittered[x, :, :] = self.color_jitter(Image.fromarray(input_data[x, :, :], mode='L'))
        return torch.Tensor(input_jittered) / 255
