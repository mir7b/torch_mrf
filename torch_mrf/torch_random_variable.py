"""This module holds definitions for easy to use random variables."""

import torch
from torch_mrf.mrf_utils import binary
class RandomVariable(object):
    """A random variable which can only be 1 value at a time.
    
    Attributes:
        name (str): The name of the variable
        domain (list<str>): The domain of this variable sorted lexical
        domain_length (int): The length of the domain
        encoding_size (int): The size of the encoding

    """

    def __init__(self, name, domain, domain_length=None, encoding_length=None):
        """Construct a random variable with a name and possible values.
        
        Args:
            name (str): The name of the variable
            domain (list<str>): The domain of this variable
            domain_length (int): The length of the domain
            encoding_size (int): The size of the encoding

        """
        self.name = name
        self.domain = sorted(domain)
        self.domain_length = len(domain) if domain_length is None else domain_length
        self.encoding_length = encoding_length if encoding_length else torch.ceil(torch.log2(torch.tensor(self.domain_length))).long().item()
        print(self.encoding_length)

    def encode(self, value):
        """Encode the value of the variable as one hot encoded tensor.
        
        Returns:
            encoding (torch.tensor<torch.bool>): The encoding of the value in its domain
        """
        return binary(torch.tensor(self.domain.index(value)),torch.tensor(self.encoding_length))

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
        super(BinaryRandomVariable, self).__init__(name, [False, True], domain_length=2, encoding_length=1)

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