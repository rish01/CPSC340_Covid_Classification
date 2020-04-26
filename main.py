from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd

import pickle

from CovidDataset import CovidDatasetTrain, CovidDatasetTest

train_imgs_path = os.path.join(os.path.abspath(__file__), '..', 'data', 'train_images_512.pk')
train_labels_path = os.path.join(os.path.abspath(__file__), '..', 'data', 'train_labels_512.pk')
test_imgs_path = os.path.join(os.path.abspath(__file__), '..', 'data', 'test_images_512.pk')

train_imgs = pickle.load(open(train_imgs_path, 'rb'), encoding='bytes')
train_labels = pickle.load(open(train_labels_path, 'rb'), encoding='bytes')
test_imgs = pickle.load(open(test_imgs_path, 'rb'), encoding='bytes')

print(type(train_imgs))
print(train_imgs.shape)
print(type(train_labels))
print(train_labels.shape)
print(type(test_imgs))
print(test_imgs.shape)


def make_data_loaders():
    train_dataset = CovidDatasetTrain(train_imgs, train_labels)
    test_dataset = CovidDatasetTest(test_imgs)

    return {
        "train": DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1),
        "test": DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1),
    }


data_loaders = make_data_loaders()
dataset_sizes = {'train': len(data_loaders['train'].dataset),
                 'test':len(data_loaders['test'].dataset)}

class_names = ['covid', 'background']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load and save the images if they haven't been saved already
training_images_rgb_folder = os.path.join(os.path.abspath(__file__), '..', 'training_images_rgb')
training_images_grayscale_folder = os.path.join(os.path.abspath(__file__), '..', 'training_images_grayscale')
test_images_rgb_folder = os.path.join(os.path.abspath(__file__), '..', 'test_images_rgb')
test_images_grayscale_folder = os.path.join(os.path.abspath(__file__), '..', 'test_images_grayscale')

if not os.listdir(training_images_rgb_folder):
    index = 0
    for sample in data_loaders["train"].dataset.imgs:
        image_name = f"Image_{index}_covid{train_labels[index].numpy()}.png"
        plt.imsave(os.path.join(training_images_rgb_folder, image_name), sample[0])
        plt.imsave(os.path.join(training_images_grayscale_folder, image_name), sample[0], cmap='gray')
        print(f"Saved {image_name}")
        index += 1

if not os.listdir(test_images_rgb_folder):
    index = 0
    for sample in data_loaders["test"].dataset.imgs:
        image_name = f"Image_{index}.png"
        plt.imsave(os.path.join(test_images_rgb_folder, image_name), sample[0])
        plt.imsave(os.path.join(test_images_grayscale_folder, image_name), sample[0], cmap='gray')
        print(f"Saved {image_name}")
        index += 1

# Training loop starter
num_epochs = 1      # Set this yourself

for epoch in range(num_epochs):
    for sample in data_loaders["train"]:
        pass
    # Image shape
    # Batch size x Channels x Width x Height
    print(sample[0].shape)
    # Labels shape
    # Batch size
    print(sample[1].shape)
