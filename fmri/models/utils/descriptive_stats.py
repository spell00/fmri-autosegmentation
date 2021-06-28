import os
from tqdm import tqdm
import torch
import numpy as np
import nibabel as nib


def get_normalized(x):
    x[torch.isnan(x)] = 0
    x = x.float()
    if x.max() > 0:
        x /= x.max()
    return x


samples = os.listdir('TrainingImages/train/images')
samples.sort()
means = []
stds = []

for idx in range(len(samples)):
    name = samples[idx]
    x = nib.load(f"TrainingImages/train/images/{name}")
    x = x.dataobj
    x = torch.Tensor(np.array(x))
    x = get_normalized(x)
    means += [x.mean().item()]
    stds += [x.std().item()]
print(f'mean: {np.mean(means)}\nstd: {np.mean(stds)}')
