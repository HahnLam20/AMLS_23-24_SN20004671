import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image as im

#sklearn imports
from sklearn.utils import shuffle

#torch imports
import torch
from torchvision.transforms import v2, transforms

def preprocess(train_images, train_labels):
    '''
    Function used to preprocess the dataset and complete data augmentation,
    as well as provide images for visualisation after augmentation. Used for
    Task A.
    :param train_images: Training images extracted from dataset
    :param train_labels: Training labels extracted from dataset
    :return: Augmented training images and labels (with shuffling)
    '''

    # data augmentation - increasing size of train dataset
    train_transforms = v2.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(28, scale=(0.9, 1.0)),
        transforms.ToTensor(),
    ])
    train_transformed = np.squeeze([train_transforms(im.fromarray(train_images[i])) for i in range(len(train_images))])
    print(train_transformed.shape)
    train_images = np.append(train_images, train_transformed, axis=0)
    train_labels = np.append(train_labels, train_labels, axis=0)
    train_labels = train_labels.ravel()
    print(train_images.shape)
    print(train_labels.shape)
    train_images, train_labels = shuffle(train_images, train_labels)
    return train_images, train_labels
