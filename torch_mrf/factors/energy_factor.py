import torch
import torch.nn as nn
from typing import List
import torch_random_variable.torch_random_variable as trv
import plotly.graph_objects as go
import tqdm
from .discrete_factor import DiscreteFactor

class EnergyFactor(DiscreteFactor):
    """
    A discrete factor function that maps every world that is possible by its random variables to a real positive number.
    
    :param random_variables: The random variables that are used by this factor.
    :type random_variables: List of torch variables
    :param verbose: The verbosity level
    :type verbose: int
    :param device: The hardware device the factor will do its calculations on
    :type device: str or int
    :param max_parallel_worlds: The maximum number of parallel worlds that will be used by this factor. Setting this parameter too
        low can take alot of time. Setting it too high can cause a memory problem.
    :type max_parallel_worlds: int
    :param weights: The weights that are used to calculate the potential of a state.
    :type weights: torch.tensor
    """
    
    def __init__(self,random_variables:List[trv.RandomVariable], device:str or int="cuda",
                 max_parallel_worlds:int = pow(2,20), fill_value = 0., verbosity:int=1):
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
    
        super(EnergyFactor, self).__init__(random_variables, device,
                 max_parallel_worlds, fill_value, verbosity)
    
    
    def fit(self, data:torch.Tensor):
        """Calculates weights for this factor by infering the probability of each sample in the provided data.
        This is optimal w. r. t. the parameters as proven in TODO.

        Args:
            data (torch.Tensor): The data that is observed for this factor. data is a tensor of shape 
                                 (number of observations ,number of random_variables in this factor)) 
                                 and and indexable type like long or bool etc.
        """
        with torch.no_grad():
            if self.verbosity <=1:
                for state in torch.cartesian_prod(*[torch.arange(var.domain_length) for var in self.random_variables]):
                    prob =  torch.all(data == state, dim=1).double().sum()
                    prob /= float(len(data))
                    self.weights[state.chunk(len(self.random_variables),-1)] = torch.log(prob)
            else:
                for state in tqdm.tqdm(torch.cartesian_prod(*[torch.arange(var.domain_length) for var in self.random_variables]),
                                       desc="Fitting Factor %s" % self.random_variables):
                    prob =  torch.all(data == state, dim=1).double().sum()
                    prob /= float(len(data))
                    self.weights[state.chunk(len(self.random_variables),-1)] = torch.log(prob)
    