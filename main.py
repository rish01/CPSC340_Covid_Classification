from __future__ import print_function, division

import winsound
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.models import resnet18, resnet34, resnet152
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
from utils import save_results_in_csv, save_images, train
from transfer_learning_model import TransferLearningModel
from FullyConnectedLinearModel import *


if __name__ == '__main__':
    # ########################## IMPORTANT INPUT - SPECIFY WHICH MODEL TO RUN ############################################ #
    model_to_run = "KERAS_CNN"      # Choose from KERAS_CNN, TRANSFER_LEARNING, FULLY_CONNECTED_LINEAR
    ########################################################################################################################

    # The data is located in the 'data' folder
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
    dataset_sizes_tl = {'train': len(data_loaders_transfer_learning['train'].dataset),
                        'val': len(data_loaders_transfer_learning['val'].dataset),
                        'test':len(data_loaders_transfer_learning['test'].dataset)}
    data_loaders_original = make_data_loaders_original()
    dataset_sizes = {'train': len(data_loaders_original['train'].dataset),
                     'test':len(data_loaders_original['test'].dataset)}


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
        keras_model.fit(X=X_train, y=y_train)

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


    elif model_to_run == "FULLY_CONNECTED_LINEAR":
        zero_label_idx = (train_labels == 0).nonzero()
        zero_imgs = train_imgs[zero_label_idx].squeeze()
        aug_train_imgs = torch.cat((train_imgs, zero_imgs))
        train_imgs = aug_train_imgs
        aug_train_labels = torch.cat((train_labels, torch.zeros(len(zero_imgs)).long()))
        train_labels = aug_train_labels

        class_names = ['covid', 'background']
        classes = len(class_names)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_data = CovidDatasetTrain(train_imgs, train_labels)
        test_data = CovidDatasetTest(test_imgs)

        dev_size = 1 / 7
        batch_size = 1

        idx = list(range(len(train_imgs)))
        np.random.shuffle(idx)
        split_size = int(np.floor(dev_size * len(train_imgs)))
        train_idx, dev_idx = idx[split_size:], idx[:split_size]
        train_sampler = SubsetRandomSampler(train_idx)
        dev_sampler = SubsetRandomSampler(dev_idx)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
        dev_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=dev_sampler)
        test_loader = torch.utils.data.DataLoader(test_imgs, batch_size=batch_size)

        dataset_sizes = {'train': int(np.ceil(len(train_loader.dataset) * (1 - dev_size))),
                         'valid': int(np.floor(len(dev_loader.dataset) * dev_size))}
        data_loaders = {'train': train_loader, 'valid': dev_loader}

        resnet_model = resnet152(pretrained=True)

        for name, p in resnet_model.named_parameters():
            if 'bn' not in name:
                p.requires_grad = False

        num_ftrs = resnet_model.fc.in_features
        fc = FullyConnectedLinearModel(num_ftrs, classes)
        resnet_model.fc = fc

        optimizer = optim.Adam(resnet_model.parameters())
        exp_learning_rate_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        model_ft = train(resnet_model, data_loaders, dataset_sizes, nn.CrossEntropyLoss(), optimizer,
                         exp_learning_rate_scheduler, num_epochs=15)

        torch.save(resnet_model, "resnet_model.pkl")

        for sample in test_loader:
            pred = model_ft(sample)
            pred = pred.detach().numpy()
            print(np.argmax(pred))

        duration = 1000  # milliseconds
        freq = 440  # Hz
        winsound.Beep(freq, duration)



