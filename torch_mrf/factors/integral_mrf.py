from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch_mrf import mrf_utils
import tqdm
import plotly.express as px
import plotly.graph_objects as go
import torch_random_variable.torch_random_variable as trv
import plotly.express as px


class IntegralFactor(nn.Module):
    def __init__(self, random_variables:List[trv.RandomVariable], device:str or int="cpu:0", max_parallel_worlds:int = pow(2,20),
                 verbose:int=1, fill_value = 0.):
        super(IntegralFactor, self).__init__()
        
        self.random_variables = random_variables 
        self.device = device
        self.verbose = verbose
        self.weights = nn.parameter.Parameter(data=torch.full(size=[var.domain_length for var in self.random_variables], 
                                            fill_value=fill_value, dtype=torch.double, device=self.device))


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        illegal_queries,_ = torch.where(x < 0)
        illegal_queries = torch.unique(illegal_queries)
        result = self.weights[x.chunk(len(self.random_variables),-1)].squeeze(-1)
        result[illegal_queries] = 0
        return result
    
    
    def discrete_probability(self, x:torch.Tensor) -> torch.Tensor:
        probability = self(x)
        for idx, random_variables in enumerate(self.random_variables):
            lower_diff = functional.one_hot(torch.full((len(x),), idx), len(self.random_variables))
            lower_bound = x - lower_diff
            probability = probability -  self(lower_bound)
        
        redundant_area = x - torch.ones(x.shape, dtype=torch.long)
        probability = probability + (len(self.random_variables) -1) * self(redundant_area)
        return probability
    
    def fit(self, data:torch.Tensor):
        """data is a tensor of shape (number of observations ,number of random_variables in this clique))"""
        with torch.no_grad():
            for state in torch.cartesian_prod(*[torch.arange(var.domain_length) for var in self.random_variables]):
                prob =  torch.all(data <= state, dim=1).double().sum()
                prob /= float(len(data))
                self.weights[state.chunk(len(self.random_variables),-1)] = prob
    
    def plot(self) -> go.Scatter:
        pass
    
class IntegralMarkovRandomField(markov_network.MarkovNetwork):
    
    def __init__(self, random_variables:List[trv.RandomVariable], cliques:List[List[Union[str, trv.RandomVariable]]], 
            device:str or int="cpu:0", max_parallel_worlds:int = pow(2,20),verbose:int=1):
        
        super(IntegralMarkovRandomField, self).__init__(random_variables, cliques, IntegralFactor, device, max_parallel_worlds,
                                                        verbose)
    
    def probability(self, lower_bound, upper_bound):
        return self(upper_bound) - self(lower_bound)

    def discrete_probability(self, x:torch.Tensor) -> torch.Tensor:
        probs = torch.ones((len(x),), device = self.device, dtype = torch.float)
        for clique in self.cliques:
            rows = self._get_rows_in_universe(clique.random_variables)
            potential = clique.discrete_probability(x[:,rows])
            probs = probs * potential
        return probs
    
    def calc_z(self, set_Z:bool = True):
        if set_Z:
            self.Z = torch.tensor(1, dtype=torch.double)
        return torch.tensor(1, dtype=torch.double)
        
    def calc_z_derivative(self):
        Z = torch.tensor(0, dtype=torch.double, device=self.device)

        #iterate over world batches
        for world_batch in mrf_utils.iter_universe_batches_(self.random_variables, max_worlds=self.max_parallel_worlds, 
                                                           verbose=self.verbose > 0):

            #calc their probability mass
            probabilities = self.discrete_probability(world_batch.to(self.device))

            #add up the new overall probability mass
            Z += torch.sum(probabilities)
            
        return Z


def exact_probability(data):
    probs = torch.zeros((len(data),))
    for idx, sample in enumerate(data):
        prob =  torch.all(data <= sample, dim=1).double().sum()
        prob /= float(len(data))
        probs[idx] = prob
    return probs

def main():
    
    a = trv.RandomVariable("Weather", [0,1,2,3])
    b = trv.RandomVariable("Mood", [0,1,2])
    c = trv.RandomVariable("Saskia", [0,1])
    

    data = torch.cat((torch.randint(0,4, (10000,)).unsqueeze(0),
                      torch.randint(0,3, (10000,)).unsqueeze(0), 
                      torch.randint(0,2, (10000,)).unsqueeze(0))).T

    imrf = IntegralMarkovRandomField([a,b,c], [[a,b], [b,c]])
    imrf.fit(data)
    
    mrf = markov_network.MarkovNetwork([a,b,c], [[a,b], [b,c]])
    mrf.fit(data)
    print(mrf.Z)
    
    px.scatter_3d(x = data[:,0], y=data[:,1], z=data[:,2]).show()
    
if __name__ == "__main__":
    main()