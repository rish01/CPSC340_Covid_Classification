import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable


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


# Reference: Mitchell, L., Subramanian, V., & K., S. Y. (2019). Deep learning with PyTorch 1.x: implement deep
# learning techniques and neural network architecture variants using Python. page 173. Birmingham: Packt Publishing.
def train(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()
    if torch.cuda.is_available():
        model.cuda()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(scheduler.get_lr())
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                # print(loss)
                running_loss += loss.item()
                print(preds, labels.data)
                print(preds.item() == labels.data.item())
                running_corrects += np.sum(int(preds.item()) == int(labels.data.item()))
                print("running_corrects", running_corrects)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print("epoch_acc running_corrects ", running_corrects)
            print("epoch_acc ", dataset_sizes[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def save_images(data_loaders, train_labels):
    """
    Load and save the images in the following folders
    :param data_loaders: contains the tensors of the image pixels
    :param train_labels: contains 1/0 labels indicating whether the lung image is of covid patient or not
    :return: 
    """
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