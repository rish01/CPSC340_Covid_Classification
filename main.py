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
import json
import pickle
from sklearn.preprocessing import LabelBinarizer

from keras_model import KerasModel
from CovidDataset import CovidDatasetTrain, CovidDatasetTest
from utils import save_results_in_csv, save_images
from transfer_learning_model import TransferLearningModel

# ########################## IMPORTANT INPUT - SPECIFY WHICH MODEL TO RUN ############################################ #
model_to_run = "KERAS_CNN"      # Choose from KERAS_CNN, TRANSFER_LEARNING
########################################################################################################################

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

#############################################
    # Data Augmentation Transformations

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ]),
}

# Split data into train and validation
def make_data_loaders_transfer_learning(batch_size=5):
    train_dataset = CovidDatasetTrain(train_imgs, train_labels, transform = data_transforms['train'])
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50, 20])
    test_dataset = CovidDatasetTest(test_imgs, transform=data_transforms['test'])

    return {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1),
        "test": DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1),

    }

def make_data_loaders_original():
    train_dataset = CovidDatasetTrain(train_imgs, train_labels)
    test_dataset = CovidDatasetTest(test_imgs)

    return {
        "train": DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1),
        "test": DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1),
    }


data_loaders_transfer_learning = make_data_loaders_transfer_learning()
dataset_sizes_tl = {'train': len(data_loaders['train'].dataset),
                    'val': len(data_loaders['val'].dataset),
                    'test':len(data_loaders['test'].dataset)}
data_loaders_original = make_data_loaders_original()
dataset_sizes = {'train': len(data_loaders_transfer_learning['train'].dataset),
                 'test':len(data_loaders_transfer_learning['test'].dataset)}

class_names = ['covid', 'background']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if model_to_run == "KERAS_CNN":
    # ################################################ KERAS MODEL ################################################### #
    # These file paths are specified so that model parameters can be saved after training
    model_name_json_path = os.path.join(os.path.abspath(__file__), '..', 'data', 'Keras_best_model.json')
    model_name_h5_path = os.path.join(os.path.abspath(__file__), '..', 'data', 'Keras_best_model.h5')

    y_train = train_labels.numpy()
    X_train = data_loaders_original["train"].dataset.imgs
    X_test = data_loaders_original["test"].dataset.imgs

    keras_model = KerasModel(model_name_json_path=model_name_json_path, model_name_h5_path=model_name_h5_path, X=X_train)
    # keras_model.fit(X=X_train, y=y_train)

    y_pred = keras_model.predict(X_train)
    tr_error = np.mean(y_pred != y_train[:, None])
    print(f"Keras Model Training Error is: {tr_error}")
    test_labels = keras_model.predict(X_test)
    save_results_in_csv(test_labels)

elif model_to_run == "TRANSFER_LEARNING":
    ### load Resnet152 pre-trained model
    model_conv = torchvision.models.resnet152(pretrained=True)

    model = TransferLearningModel(model_conv)

    #Train the model with pre-trained Resnet
    print("Training model...")
    model_conv = model.fit(data_loaders_transfer_learning, dataset_sizes_tl, num_epochs=40)
    print("Model Training Done")

    # Make predictions for test data
    print("Making predictions on test data...")
    pred = model.predict(data_loaders_transfer_learning, dataset_sizes_tl)

    # save the predictions for submission
    save_results_in_csv(pred)
    print('predictions saved and ready for submission')
