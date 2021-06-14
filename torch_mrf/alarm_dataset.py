"""This module holds implementations of Datasets which can be used for MRFs."""

import torch
from torch_mrf.torch_random_variable import BinaryRandomVariable, RandomVariable

class AlarmDataset(torch.utils.data.Dataset):
    """A dataset for a Markov Random Field (MRF) with which the mrf can be tested and debugged.

    Attributes:
        samples (torch.Tensor<torch.bool>): A tensor of complete observations. 
    """

    def __init__(self, num_samples=10000):
        """Construct an Alarm Dataset with n samples.
        
        Args:
            num_samples (int): The number of samples.
        """

        super(AlarmDataset, self).__init__()
        self.random_variables = []
        self.random_variables.append(RandomVariable(name="Burglary", domain=[True, False]))
        self.random_variables.append(BinaryRandomVariable(name="Earthquake"))
        self.random_variables.append(BinaryRandomVariable(name="Alarm"))
        self.random_variables.append(BinaryRandomVariable(name="MaryCalls"))
        self.random_variables.append(BinaryRandomVariable(name="JohnCalls"))
        self.random_variables.sort(key=lambda x: x.name)

        self.samples = torch.randint(low=0, high=2, size=(num_samples,len(self.random_variables))).bool()

    def __len__(self):
        """Return the number of samples in this dataset.
        
        Returns:
            length (int): The amount of samples in this dataset.
            
        """
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]