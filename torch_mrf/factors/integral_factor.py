from typing import List
import torch
import torch.nn as nn
import torch_random_variable.torch_random_variable as trv
from .discrete_factor import DiscreteFactor
import tqdm

class DiscreteIntegralFactor(DiscreteFactor):
    def __init__(self,random_variables:List[trv.RandomVariable], device:str or int="cuda",
                 max_parallel_worlds:int = pow(2,20), fill_value = 1., verbosity:int=1):
        """Create a factor that describes the potential of each state. This factor has exponential many parameters in the
        number of random variables.

        Args:
            random_variables (List[trv.RandomVariable]): The involved random variables.
            device (str or int, optional): Device the factor will lay on. Defaults to "cuda".
            max_parallel_worlds (int, optional): The maximum number of parallel worlds that will be used by this factor. 
                Setting this parameter too low can take alot of time. Setting it too high can cause a memory problem. 
                Defaults to pow(2,20).
            fill_value ([type], optional): The default value for each weight in the factor. Defaults to 1..
            verbose (int, optional): The verbosity level of this factor. Defaults to 1.
        """
    
        super(DiscreteIntegralFactor, self).__init__(random_variables, device,
                 max_parallel_worlds, fill_value, verbosity)
        
    
    def fit(self, data:torch.Tensor):
        """data is a tensor of shape (number of observations ,number of random_variables in this clique))"""
        with torch.no_grad():
            if self.verbosity <= 1:
                for state in torch.cartesian_prod(*[torch.arange(var.domain_length) for var in self.random_variables]):
                    prob =  torch.all(data <= state, dim=1).double().sum()
                    prob /= float(len(data))
                    self.weights[state.chunk(len(self.random_variables),-1)] = prob
            else:
                for state in tqdm.tqdm(torch.cartesian_prod(*[torch.arange(var.domain_length) for var in self.random_variables]),
                                       desc="Fitting Factor %s" % self.random_variables):
                    prob =  torch.all(data <= state, dim=1).double().sum()
                    prob /= float(len(data))
                    self.weights[state.chunk(len(self.random_variables),-1)] = prob 
    

class ContinuousIntegralFactor(nn.Module):
    def __init__(self,random_variables:List[trv.RandomVariable], device:str or int="cuda",
                 max_parallel_worlds:int = pow(2,20), model=None, verbosity:int=1):
        """Create a factor that describes the potential of each state. This factor has exponential many parameters in the
        number of random variables.

        Args:
            random_variables (List[trv.RandomVariable]): The involved random variables.
            device (str or int, optional): Device the factor will lay on. Defaults to "cuda".
            max_parallel_worlds (int, optional): The maximum number of parallel worlds that will be used by this factor. 
                Setting this parameter too low can take alot of time. Setting it too high can cause a memory problem. 
                Defaults to pow(2,20).
            model (nn.Sequential, optional): A torch model that can be trained to represent the joint probability distribution
            verbose (int, optional): The verbosity level of this factor. Defaults to 1.
        """
    
        super(ContinuousIntegralFactor, self).__init__()
        
        self.random_variables:List[trv.RandomVariable] = random_variables
        self.verbosity:int = verbosity
        self.device:str or int = device
        self.max_parallel_worlds:int = max_parallel_worlds
        self.model = model
        
    
    def fit(self, data:torch.Tensor):
        """data is a tensor of shape (number of observations ,number of random_variables in this clique))"""
        