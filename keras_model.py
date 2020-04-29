import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, BatchNormalization
from keras import metrics
import json

from CovidDataset import CovidDatasetTrain, CovidDatasetTest
from COVID_Images_Sequence import COVIDImagesSequence
from COVID_Images_Callback import Metrics

# ################################################ KERAS MODEL ##################################################### #


class KerasModel:
    """
    This model replicates the CNN architecture present on slide 5 (ImageNet Insights) of lecture 35b.
    Several inputs models were trialed but the ones present below yielded a training error of 15.71%
    """

    def __init__(self, model_name_json_path, model_name_h5_path, X):
        self.model_name_json_path = model_name_json_path
        self.model_name_h5_path = model_name_h5_path

        # Computation of values to be used for zero mean and normalizing the data
        self.dataset_min = torch.min(-X)
        self.dataset_max = torch.max(-X)
        self.dataset_range = self.dataset_max - self.dataset_min

    def fit(self, X, y):
        X = torch.div(torch.add(-X, -self.dataset_min), self.dataset_range)
        X = torch.add(X, -0.5)

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

        model.validation_data = (X, y)

        model.fit(X, y, batch_size=10, epochs=5, verbose=1, callbacks=[Metrics()])
        model.save_weights(self.model_name_h5_path)

        model_json = model.to_json()
        with open(self.model_name_json_path, "w") as json_file:
            json.dump(model_json, json_file)

        model.save_weights(self.model_name_h5_path)

    def predict(self, X):
        X = torch.div(torch.add(-X, -self.dataset_min), self.dataset_range)
        X = torch.add(X, -0.5)

        try:
            with open(self.model_name_json_path, 'r') as f:
                model_json = json.load(f)
        except:
            print("Error: Model's json file is not present. Please run fit function before running predict.")
        else:
            model = model_from_json(model_json)
            model.load_weights(self.model_name_h5_path)
            y_pred = model.predict_classes(X, verbose=0)
            return y_pred


