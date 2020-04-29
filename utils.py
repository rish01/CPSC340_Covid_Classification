import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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