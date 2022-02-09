"""This module holds definitions for easy to use random variables."""

import torch
from .utils import binary, decimal
import itertools

class RandomVariable(object):
    """A random variable which can only be 1 value at a time.
    
    Attributes:
        name (str): The name of the variable
        domain (list<str>): The domain of this variable sorted lexical
        domain_length (int): The length of the domain
        encoding_size (int): The size of the encoding

    """

    def __init__(self, name, domain):
        """Construct a random variable with a name and possible values.
        
        Args:
            name (str): The name of the variable
            domain (list<str>): The domain of this variable

        """
        self.name = name
        self.domain = sorted(domain)
        self.domain_length = len(domain)
        self.encoding_length = torch.ceil(torch.log2(torch.tensor(self.domain_length))).long().item()

    def encode(self, value):
        """Encode the value of the variable as one hot encoded tensor.
        
        Returns:
            encoding (torch.tensor<torch.bool>): The encoding of the value in its domain
        """
        return binary(torch.tensor(self.domain.index(value)),torch.tensor(self.encoding_length)).bool()
    
    def decode(self, bits):
        """Decode a sequence of bits to its domain value.
          
        Returns:
            value: The value of the domain that was encoded
        """
        return [self.domain[idx] for idx in decimal(bits, torch.tensor(self.encoding_length))]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name
    
    def __ne__(self, other):
        return not (self == other)
    
    def __len__(self):
        return len(self.domain)

class MultiDomainRandomVariable(RandomVariable):
    """A random variable with a domain that consists of the cartesian product
        of multiple domains. 
    """
    
    def __init__(self, name, domains):
        super(MultiDomainRandomVariable, self).__init__(name, list(itertools.product(*domains)))


class BinaryRandomVariable(RandomVariable):
    """A binary random variable.
    
    It benefits from only half the encoding space required as Random Variable with a domain of [True,False]
    """

    def __init__(self, name):
        """Construct a binary variable from a name.
        
        Args:
            name (str): The name of the variable
        """
        super(BinaryRandomVariable, self).__init__(name, [False, True])

    def __str__(self):
        return "Torch Binary Random Variable " + self.name


class NumericRandomVariable(RandomVariable):
    """A numeric random variable which creates a discretization of a numeric domain
        that maximizes the likelihood."""


    def __init__(self, name, domain):
        """Construct a numeric variable from a name.
        
        Args:
            name (str): The name of the variable
            domain (list): A list of intervals that partition the variable space
        """
        self.name = name
        self.domain = domain
        super(NumericRandomVariable, self).__init__(name, domain)

    def encode(self, value):
        interval_idx = 0
        for idx, interval in enumerate(self.domain):
            if value in interval:
                interval_idx = idx
                break
        
        return binary(torch.tensor(interval_idx),torch.tensor(self.encoding_length)).bool()

    def __str__(self):
        return self.name


class Interval:
    def __init__(self, lower_bound:float = -float("inf"), upper_bound:float = float("inf")):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __eq__(self, o: object) -> bool:
        return self.lower_bound == o.lower_bound and self.upper_bound == o.upper_bound

    def __contains__(self, item:float):
        return item >= self.lower_bound and item < self.upper_bound
    
    def __repr__(self):
        return str(self)

    def __str__(self):
        return "Interval [%s, %s]" % (self.lower_bound, self.upper_bound)
    
    def __lt__(self, other):
        return self.lower_bound < other.lower_bound and self.upper_bound < other.upper_bound
    
    def __leq__(self, other):
        return self.lower_bound <= other.lower_bound and self.upper_bound <= other.upper_bound

    def __gt__(self, other):
        return self.lower_bound > other.lower_bound and self.upper_bound > other.upper_bound
    
    def __geq__(self, other):
        return self.lower_bound >= other.lower_bound and self.upper >= other.upper_bound