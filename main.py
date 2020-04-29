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

from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, BatchNormalization
from keras import metrics
from keras.utils import np_utils

import pickle

from CovidDataset import CovidDatasetTrain, CovidDatasetTest
from COVID_Images_Sequence import COVIDImagesSequence
from COVID_Images_Callback import Metrics

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


def save_results_in_csv(y_pred):
    '''
    Saves the predictions of test lung images in the provided csv file
    :param y_pred: numpy array containing the boolean predictions
    '''
    fname = os.path.join(os.path.abspath(__file__), '..', 'data', 'CPSC340_Q2_SUBMISSION.csv')
    y_pred = y_pred.astype(bool)

    df = pd.DataFrame(columns=['Id', 'Predicted'])
    df['Id'] = np.arange(0, y_pred.shape[0])
    df['Predicted'] = y_pred
    df.to_csv(fname, index=False)


data_loaders = make_data_loaders()
dataset_sizes = {'train': len(data_loaders['train'].dataset),
                 'test':len(data_loaders['test'].dataset)}

# Normalize the provided data to [0, 1]
dataset_min = torch.min(-data_loaders["train"].dataset.imgs)
dataset_max = torch.max(-data_loaders["train"].dataset.imgs)
dataset_range = dataset_max - dataset_min

data_loaders["train"].dataset.imgs = torch.div(torch.add(-data_loaders["train"].dataset.imgs, -dataset_min), dataset_range)
data_loaders["train"].dataset.imgs = torch.add(data_loaders["train"].dataset.imgs, -0.5)

data_loaders["test"].dataset.imgs = torch.div(torch.add(-data_loaders["test"].dataset.imgs, -dataset_min), dataset_range)
data_loaders["test"].dataset.imgs = torch.add(data_loaders["test"].dataset.imgs, -0.5)

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



# ################################################ KERAS MODEL ##################################################### #

# This model replicates the CNN architecture present on slide 5 (ImageNet Insights) of lecture 35b.
# Several inputs models were trialed but the ones present below yielded a training error of 15.71%
y_train = train_labels.numpy()
X_train = data_loaders["train"].dataset.imgs.numpy()
X_test = data_loaders["test"].dataset.imgs
training = False
predicting = True
model_json_file = os.path.join(os.path.abspath(__file__), '..', 'data', 'Keras_best_model.json')
model_name = os.path.join(os.path.abspath(__file__), '..', 'data', 'Keras_best_model.h5')

if training:
    print("Starting Keras Neural Network Training!")
    model = Sequential()

    model.add(Conv2D(64, (5, 5), input_shape=(3, 512, 512), data_format='channels_first', padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(1, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[metrics.binary_accuracy])

    model.validation_data = (X_train, y_train)


    model.fit(X_train, y_train, batch_size=10, epochs=5, verbose=1, callbacks=[Metrics()])
    # model.fit(x=COVIDImagesSequence(data_loaders["train"].dataset.imgs.numpy(), train_labels.numpy(), 10),
              # epochs=5, verbose=1)
    model.save_weights(model_name)

    model_json = model.to_json()
    with open(model_json_file, "w") as json_file:
        json.dump(model_json, json_file)

    model.save_weights(model_name)

if predicting:
    with open(model_json_file, 'r') as f:
        model_json = json.load(f)

    model = model_from_json(model_json)
    model.load_weights(model_name)
    y_pred = model.predict_classes(X_train, verbose=0)
    tr_error = np.mean(y_pred != y_train[:, None])
    print(f"Keras Model Training Error is: {tr_error}")
    test_labels = model.predict_classes(X_test, verbose=0)
    save_results_in_csv(test_labels)
