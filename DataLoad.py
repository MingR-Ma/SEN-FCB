from torch.utils.data import Dataset
import numpy as np
import torch
from random import shuffle
import itertools


class BrainDataGenerator(Dataset):
    def __init__(self, train_names, mode='train'):
        """
        :param json_file:
        :param h5_file:
        """
        self.mode = mode

        self.pair = list(itertools.permutations(train_names, 2))
        shuffle(self.pair)

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, index):
        self.data_A = np.load(self.pair[index][0])
        self.data_B = np.load(self.pair[index][1])

        return torch.Tensor(self.data_A),torch.Tensor(self.data_B)


class CardiacDataGenerator(Dataset):
    def __init__(self, train_names,large_path,small_path, mode='train'):

        self.mode = mode
        self.train_names=train_names
        self.large_path=large_path
        self.small_path=small_path

        shuffle(self.train_names)

    def __len__(self):
        return len(self.train_names)

    def __getitem__(self, index):
        # small->large
        # print(self.train_names[index])
        self.data_A = np.load(self.small_path+f'{self.train_names[index]}_HumanSmall.npy')
        self.data_B = np.load(self.large_path+f'{self.train_names[index]}_HumanLarge.npy')

        return torch.Tensor(self.data_A),torch.Tensor(self.data_B)
