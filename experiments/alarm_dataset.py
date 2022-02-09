"""This module holds implementations of Datasets which can be used for MRFs."""

import torch
from torch_random_variable.torch_random_variable import BinaryRandomVariable, RandomVariable
import tqdm

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

        burglary = torch.bernoulli(torch.full(size=(num_samples,), fill_value=0.1))
        earthquake = torch.bernoulli(torch.full(size=(num_samples,), fill_value=0.2))

        
        alarm = torch.full(size=(num_samples,), fill_value=0.)
        for idx, row in enumerate(tqdm.tqdm(torch.cat((burglary.unsqueeze(dim=0),earthquake.unsqueeze(dim=0)), dim=0).T, 
                                  desc="Sampling Data")):
            if torch.all(row == torch.tensor([1.,1.])):
                alarm[idx] = 0.95
            elif torch.all(row == torch.tensor([1.,0.])):
                alarm[idx] = 0.94
            elif torch.all(row == torch.tensor([0.,1.])):
                alarm[idx] = 0.29
            elif torch.all(row == torch.tensor([0.,0.])):
                alarm[idx] = 0.1

        alarm = torch.bernoulli(alarm)
        
        john_calls = torch.bernoulli(torch.where(alarm==1., 0.9, 0.05))
        mary_calls = torch.bernoulli(torch.where(alarm==1., 0.7, 0.1))

    
        self.samples = torch.cat((alarm.unsqueeze(0), burglary.unsqueeze(0), earthquake.unsqueeze(0), john_calls.unsqueeze(0), 
                                  mary_calls.unsqueeze(0))).bool().T


    def __len__(self):
        """Return the number of samples in this dataset.
        
        Returns:
            length (int): The amount of samples in this dataset.
            
        """
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]