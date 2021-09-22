import torch_mrf.mrf
import torch
from copy import deepcopy
import networkx
import torch_mrf.mrf_utils as mrf_utils

class Trainer:
    """A trainer that can learn the structure and weights for a Markov Random Field."""

    def __init__(self, mrf:torch_mrf.mrf.MarkovRandomField, dataloader:torch.utils.data.DataLoader, 
                split_numeric_random_variables:bool = False, learn_structure:bool = False, 
                min_likelihood_improvement:bool=0.1,):
        """Initialize the trainer for an mrf.
        
        Args:
            split_numeric_random_variables (bool): Whether to learn a discretization of the numeric random variables or 
            use the existing
            learn_structure (bool): Whether to learn the cliques (structure) of the mrf or to use the exisiting
            min_likelihood_improvement (float): The minimum likelihood improvement to justify a more complex mrf
        """
        self.split_numeric_random_variables = split_numeric_random_variables
        self.learn_structure = learn_structure
        self.min_likelihood_improvement = min_likelihood_improvement
        self.mrf = mrf
        self.dataloader = dataloader

    
    def train(self):
        self.mrf.fit(self.dataloader)
        self.mrf.calc_z()

        if self.learn_structure:
            self.simplify_structure(self.mrf)

        return self.mrf

    def rate_mrf(self, mrf=None):
        mrf = mrf or self.mrf
        score = torch.tensor(0.)
        for batch in self.dataloader:
            probs = mrf(batch).cpu()
            score += sum(probs)
        return score/len(self.dataloader.dataset)
    
    def simplify_structure(self, mrf:torch_mrf.mrf.MarkovRandomField):
        graph = mrf_utils.mrf_to_networkx(mrf)
        dependencies = graph.edges

        original_mrf_score = self.rate_mrf()

        for dependency in dependencies:
            graph.remove_edge(*dependency)
            cliques = list(networkx.algorithms.clique.find_cliques(graph))
            simpler_mrf = torch_mrf.mrf.MarkovRandomField(mrf.random_variables, cliques)
            simpler_mrf.fit(self.dataloader)
            simpler_mrf.calc_z()
            simpler_mrf_score = self.rate_mrf(simpler_mrf)

            if simpler_mrf_score - original_mrf_score < self.min_likelihood_improvement:
                self.mrf = simpler_mrf