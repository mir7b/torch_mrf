from typing import List, Union
import torch
import torch.nn as nn
from torch_mrf import mrf_utils
import tqdm
import math
import frozenlist
import plotly.express as px
import plotly.graph_objects as go
import networkx
import torch_random_variable.torch_random_variable as trv

class CliqueFunction(nn.Module):
    def __init__(self, random_variables:List[trv.RandomVariable], device:str or int="cuda", max_parallel_worlds:int = pow(2,20),
                 verbose:int=1, fill_value = 0.):
        super(CliqueFunction, self).__init__()
        
        self.random_variables = random_variables
        self.device = device
        
        self.weights = nn.parameter.Parameter(data=torch.full(size=[var.domain_length for var in self.random_variables], 
                                            fill_value=fill_value, dtype=torch.double, device=self.device))


    def forward(self, x:torch.Tensor):
        return self.weights[x.chunk(len(self.random_variables),-1)]
    
    
    def fit(self, data:torch.Tensor):
        """data is a tensor of shape (number of observations ,number of random_variables in this clique))"""

        for state in torch.cartesian_prod(*[torch.arange(var.domain_length) for var in self.random_variables]):
            pass
    
class IntegralMarkovRandomField(nn.Module):
    
    def __init__(self, random_variables:List[trv.RandomVariable], cliques:List[List[Union[str, trv.RandomVariable]]], 
            device:str or int="cuda", max_parallel_worlds:int = pow(2,20),verbose:int=1):
        
        super(IntegralMarkovRandomField, self).__init__()

        #sort random variables
        self.random_variables:List[trv.RandomVariable] = random_variables
        self.random_variables.sort(key=lambda x: x.name)
        self.verbose:int = verbose
        self.device:str or int = device
        self.max_parallel_worlds:int = max_parallel_worlds

        #parse clique members to variable if they arent already variables
        for idx, clique in enumerate(cliques):
            for jdx, partner in enumerate(clique):
                if isinstance(partner, str):
                    corresponding_random_variables = [var for var in self.random_variables if var.name==partner]
                    #check if the partner is part of this mrfs random variables
                    if len(corresponding_random_variables) == 0:
                        raise Exception("Random variable name %s was used in a clique, but it does not exist in\
                                        the MRF. \n Random Variable names in this MRF: %s"\
                                        % (partner, [var.name for var in self.random_variables]))

                    cliques[idx][jdx] = corresponding_random_variables[0]
                    
        #sort the cliques and convert them to hashable datatypes
        self.cliques = [sorted(clique, key=lambda x: x.name) for clique in cliques]
        self.cliques = [frozenlist.FrozenList(clique) for clique in self.cliques]
        for clique in self.cliques:
            clique.freeze()


def main():
    
    a = trv.RandomVariable("Weather", ["Sunny", "Rainy", "Foggy"])
    b = trv.RandomVariable("Mood", ["Good", "Bad"])
    
    phi_1 = CliqueFunction([a,b])
    print(phi_1(torch.tensor([[0,0], [0,1]])))
    phi_1.fit(0)

if __name__ == "__main__":
    main()