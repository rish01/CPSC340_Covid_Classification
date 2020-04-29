# import dependencies

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

# our code
import utils


# Training and Prediction

# Code obtained and modified from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

class TransferLearningModel:

    def __init__(self, model, criterion='CrossEntropyLoss', optimizer='sgd', lr=1., momentum=0.9, decay_frequency=7, gamma=0.1):
        '''
        Initialize transfer leaning model

        criterion: ['CrossEntropyLoss', 'NegativeLogLoss']
            - The chosen loss function for training

        optimizer: ['sgd', 'adam']
            - The chosen gradient descent method

        lr: the starting learning rate for the model

        momentum: if 'sgd' is chosen as optimizer, the momentum value when doing SGD

        decay_frequency: number of epochs before the learning rate is changed by value gamma

        gamma: factor by which to change learning rate

        '''

        self.model = model

        # freeze the conv layers and pass through final linear layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = self.model.fc.in_features

        # Define final linear dense layer
        self.model.fc = nn.Linear(num_ftrs, 2)

        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.decay_frequency = decay_frequency
        self.gamma = gamma


    def fit(self, data_loaders, dataset_sizes, num_epochs=25):
        '''
        Given the model, his function fits the training and validation data,
        predicts labels for both and calculates loss and accuracy

        requires data_loaders, dataset_sizes and number of epochs to train
        '''
        # Criterion for loss function
        if self.criterion == 'NegativeLogLoss':
            criterion = nn.NLLLoss()
        elif self.criterion == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        else:
            print("Invalid criterion")
            return

        # Optimize parameters of final layers
        if self.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.fc.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer == 'adam':
            optimizer= optim.Adam(self.model.fc.parameters())
        else:
            print("Invalid optimizer")
            return

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.decay_frequency, gamma=self.gamma)

        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in data_loaders[phase]:
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the self.model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model


    def predict(self, data_loaders, dataset_sizes):
        '''
        Function to predict labels for test data
        '''
        arr = np.zeros(dataset_sizes['test'])
        self.model.eval()
        for i, inputs in enumerate(data_loaders['test']):
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            if torch.eq(preds, torch.tensor(1)):
                arr[i] = 1
        return arr.reshape((arr.shape[0],1))
