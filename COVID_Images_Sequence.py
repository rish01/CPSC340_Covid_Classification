from keras.utils import Sequence
import numpy as np


class COVIDImagesSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.epoch = 0
        self.batch_size = batch_size
        self.samples_per_label = int(self.batch_size / 2)
        self.covid1_indices = np.random.choice(np.argwhere(self.y == 1)[:, 0], size=self.samples_per_label,
                                               replace=False)
        self.covid0_indices = np.random.choice(np.argwhere(self.y == 0)[:, 0], size=self.samples_per_label,
                                               replace=False)

    def __len__(self):
        return int(np.ceil(len(self.y) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Return randomly chosen rows from both covid and non-covid lung images. Half of images per batch are covid
        and the other half, non-covid
        :param idx: default input parameter of this dunder method
        :return: batch_x: Training set containing self.batch_size examples with half being covid and other half, non-covid
        :return: batch_y: Training set labels containing self.batch_size examples with half being covid and other half, non-covid
        """

        self.covid1_indices = np.random.choice(np.argwhere(self.y == 1)[:, 0], size=self.samples_per_label, replace=False)
        self.covid0_indices = np.random.choice(np.argwhere(self.y == 0)[:, 0], size=self.samples_per_label, replace=False)

        batch_x = np.concatenate((self.x[self.covid0_indices], self.x[self.covid1_indices]), axis=0)
        batch_y = np.concatenate((self.y[self.covid0_indices], self.y[self.covid1_indices]))

        return batch_x, batch_y




