"""This module holds definitions for easy to use random variables."""

import torch

class RandomVariable(object):
    """A random variable which can only be 1 value at a time.
    
    Attributes:
        name (str): The name of the variable
        domain (list<str>): The domain of this variable sorted lexical
        domain_length (int): The length of the domain

    """

    def __init__(self, name, domain, domain_length=None):
        """Construct a random variable with a name and possible values.
        
        Args:
            name (str): The name of the variable
            domain (list<str>): The domain of this variable
            domain_length (int): The length of the domain

        """
        self.name = name
        self.domain = sorted(domain)
        self.domain_length = len(domain) if domain_length is None else domain_length

    def encode(self, value):
        """Encode the value of the variable as one hot encoded tensor.
        
        Returns:
            encoding (torch.tensor<torch.bool>): The encoding of the value in its domain
        """
        encoding =  torch.zeros(size=(self.domain_length,), dtype=torch.bool)
        encoding[self.domain.index(value)] = True
        return encoding

    def __repr__(self) -> str:
        return str(self)

    def __str__(self):
        return "Torch Random Variable " + self.name + " with domain " + str(self.domain)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name
    
    def __ne__(self, other):
        return not (self == other)


class BinaryRandomVariable(RandomVariable):
    """A binary random variable.
    
    It benefits from only half the encoding space required as Random Variable with a domain of [True,False]
    """

    def __init__(self, name):
        """Construct a binary variable from a name.
        
        Args:
            name (str): The name of the variable
        """
        super(BinaryRandomVariable, self).__init__(name, [True, False], domain_length=1)

    def encode(self,value):
        """Encode the value into the variable domain.
        
        Args:
            value(bool): Rather if the variable is True of False
        
        Returns:
            encoding (torch.Tensor<torch.bool>): The encoded value
        """
        return torch.tensor(value, dtype=torch.bool)

    def __str__(self):
        return "Torch Binary Random Variable " + self.name