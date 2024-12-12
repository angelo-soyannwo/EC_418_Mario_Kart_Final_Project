# Source: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
import numpy as np
import random
from torchvision import transforms as T
from torchvision.transforms import functional as F


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, *args):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            args = tuple(np.array([-point[0], point[1]], dtype=point.dtype) for point in args)
        return (image,) + args

class Grayscale:
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, image, *args):
        # Convert the image to grayscale
        image = F.rgb_to_grayscale(image, num_output_channels=self.num_output_channels)
        return (image,) + args


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, *args):
        for t in self.transforms:
            image, *args = t(image, *args)
        return (image,) + tuple(args)


class ColorJitter(T.ColorJitter):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args

"""
class ToTensor(object):
    def __call__(self, image, *args):
        return (F.to_tensor(image),) + args
"""

class ToTensor:
    def __call__(self, image, *args):
        # Convert the image to a PyTorch tensor
        image_tensor = F.to_tensor(image)
        return (image_tensor,) + args

