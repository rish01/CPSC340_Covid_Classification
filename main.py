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
from sklearn.decomposition import SparsePCA, PCA
import skimage.io
import skimage.restoration
import skimage.exposure
import cv2.cv2 as cv2

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
images_folder = os.path.join(os.path.abspath(__file__), '..', 'images')
grayscale_images_folder = os.path.join(os.path.abspath(__file__), '..', 'grayscale_images')

if not os.listdir(images_folder):
    index = 0
    for sample in data_loaders["train"].dataset.imgs:
        image_name = f"Image_{index}.png"
        plt.imsave(os.path.join(images_folder, image_name), sample[0])
        print(f"Saved {image_name}")
        index += 1

if not os.listdir(grayscale_images_folder):
    index = 0
    for sample in data_loaders["train"].dataset.imgs:
        image_name = f"Image_{index}.png"
        plt.imsave(os.path.join(grayscale_images_folder, image_name), sample[0], cmap='gray')
        print(f"Saved {image_name}")
        index += 1

image_5_gray = os.path.join(grayscale_images_folder, 'Image_5.png')
img = cv2.imread(image_5_gray)
mask = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)[1][:,:,0]
dst = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

# # Remove text from images
# X = data_loaders["train"].dataset.imgs[5][0]
# sparsePCA_model = SparsePCA(n_components=15, alpha=0.01)
# Z_sparse = sparsePCA_model.fit_transform(X)
# W_sparse = sparsePCA_model.components_
# X_recons_sparse = Z_sparse@W_sparse
# plt.imshow(X_recons_sparse)
#
# pca_model = PCA(n_components=5)
# Z = pca_model.fit_transform(X)
# W = pca_model.components_
# X_recons = Z@W
#
# print("Debug")


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
