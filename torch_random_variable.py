"""This module holds definitions for easy to use random variables."""

import torch

class RandomVariable(object):
    """A random variable which can only be 1 value at a time.
    
    Attributes:
        name (str): The name of the variable
        domain (list<str>): The domain of this variable sorted lexical
    """
    def __init__(self, name, domain):
        """Construct a random variable with a name and possible values.
        
        Args:
            name (str): The name of the variable
            domain (list<str>): The domain of this variable

        """
        self.name = name
        self.domain = sorted(domain)

    def encode(self, value):
        """Encode the value of the variable as one hot encoded tensor.
        
        Returns:
            encoding (torch.tensor<torch.bool>): The encoding of the value in its domain
        """
        encoding =  torch.zeros(size=(len(self.domain),), dtype=torch.bool)
        encoding[self.domain.index(value)] = True
        return encoding

    def __repr__(self) -> str:
        return str(self)

    def __str__(self):
        return "Torch Random Variable " + self.name + " with domain " + str(self.domain)