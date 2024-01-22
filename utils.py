import numpy as np
from torchvision import transforms


def norm_transform(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def inv_norm_transform(mean, std):
    return transforms.Compose([
        transforms.Normalize([0., 0., 0.], 1/std),
        transforms.Normalize(-mean, [1., 1., 1.])
    ])


def norm_by_channel(x):
    assert x.ndim == 3, "Image dim needs to be (C, Y, X)"
    x_normed = np.zeros_like(x)
    for i, chan in enumerate(x):
        x_normed[i] = (chan-chan.min())/(chan.max()-chan.min())
    return x_normed
