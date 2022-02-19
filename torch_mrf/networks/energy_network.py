from .markov_network import MarkovNetwork
from typing import List, Union
import torch
import torch.nn as nn
from torch_mrf import mrf_utils
import tqdm
import plotly.express as px
import plotly.graph_objects as go
import networkx
import itertools
import torch_random_variable.torch_random_variable as trv
from ..factors.energy_factor import EnergyFactor

class EnergyMarkovNetwork(MarkovNetwork):
    def __init__(self, random_variables:List[trv.RandomVariable], cliques:List[List[Union[str, trv.RandomVariable]]],
                factor = EnergyFactor, device:str or int="cuda", max_parallel_worlds:int = pow(2,20),verbose:int=1):
        """Construct an Energy Markov Random Field from the nodes and edges where the forward pass uses + as operator and not *.

        Args:
            random_variables (iterable<torch_random_variable.RandomVariable>): The random variables that are represented
                in this Markov Random Field
            cliques (iterable<iterable<torch_random_variable.RandomVariable>>): The connectivity of the random variables
                as a list of lists where the inner lists represent all members of a clique
            device (str): The device where the Markov Random Field will perform most of its calculations
            max_parallel_worlds (int): The maximum number of worlds that are evaluated at once on the graphics card.
                This parameter can produce an Cuda Out of Memory Error when picked too large and slow down speed when picked too small.
            verbose (int): Level of verbosity

        """
        super(EnergyMarkovNetwork, self).__init__(random_variables, cliques,
                factor, device, max_parallel_worlds, verbose)
        
    def forward_no_z(self, x):
        probs = torch.zeros((len(x),), device = self.device, dtype = torch.float)
        for clique in self.cliques:
            rows = self._get_rows_in_universe(clique.random_variables)
            potential = clique(x[:,rows])
            probs = probs + potential
        return torch.exp(probs)
     