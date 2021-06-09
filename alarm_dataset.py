"""This module holds implementations of Datasets which can be used for MRFs."""

from itertools import product
import torch
import os
import numpy as np
from torch_random_variable import BinaryRandomVariable

class AlarmDataset(torch.utils.data.Dataset):
    """A dataset for a Markov Random Field (MRF) with which the mrf can be tested and debugged.

    Attributes:
        samples: A list of lists where the outer list describes the number of samples and
        the inner lists contain the grounded atoms which describe the samples. 
    """

    def __init__(self, num_samples=10000):
        super(AlarmDataset, self).__init__()
        self.random_variables = []
        self.random_variables.append(BinaryRandomVariable(name="Burglary"))
        self.random_variables.append(BinaryRandomVariable(name="Earthquake"))
        self.random_variables.append(BinaryRandomVariable(name="Alarm"))
        self.random_variables.append(BinaryRandomVariable(name="MaryCalls"))
        self.random_variables.append(BinaryRandomVariable(name="JohnCalls"))
        self.random_variables.sort(key=lambda x: x.name)

        self.samples = torch.randint(low=0, high=2, size=(1000,2*len(self.random_variables))).bool()

    def __len__(self):
        """Return the number of samples in this dataset.
        
        Returns:
            length (int): The amount of samples in this dataset.
            
        """
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]