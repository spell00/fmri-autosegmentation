import os
import itertools
import pydicom
from pydicom import dcmread


def _resize_data(data, new_size=(160, 160, 160)):
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


def load_scan(path):
    """
    Loads scans from a folder and into a list.

    Parameters: path (Folder path)

    Returns: slices (List of slices)
    """
    slices = [dcmread(path + '/' + s) for s in os.listdir(path)]
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


import torch
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    size = 64
    basedir = '/run/media/simon/DATA&STUFF/data/'
    group = 'train'
    new_data_dir = basedir + '/' + group + '_' + str(size) + 'x' + str(size) + '/'
    if new_data_dir not in '/'.join([basedir, group]):
        os.mkdir(new_data_dir)
    data_dir = '/run/media/simon/DATA&STUFF/data/' + group + '/'

    ct_folders = os.listdir(data_dir)

    pbar = tqdm(total=len(ct_folders))
    bad = 0
    for i, fname in enumerate(ct_folders):
        try:
            x = get_pixels_hu(load_scan(data_dir + fname))
        except:
            bad += 1
            print('Bad:', bad)
            continue
        x = np.array(x)
        x = _resize_data(x, (size, size, size))
        x = torch.Tensor(x)
        torch.save(x, new_data_dir + fname)
        pbar.update(1)
