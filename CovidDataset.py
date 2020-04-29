#import dependencies
from torch.utils.data import Dataset, DataLoader


# Class includes transforms (given)
class CovidDatasetTrain(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.imgs[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.labels[idx]


class CovidDatasetTest(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        sample = self.imgs[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
