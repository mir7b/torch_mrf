import torch_mrf.mrf as mrf
import torch
from copy import deepcopy

class Trainer:
    """A trainer that can learn the structure and weights for a Markov Random Field."""

    def __init__(self, split_numeric_random_variables:bool = False, learn_structure:bool = False, min_likelihood_improvement:bool=0.1,
                batch_size = 1024):
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
        self.batch_size = batch_size

    
    def train(self, mrf:mrf.MarkovRandomField, dataloader:torch.utils.data.DataLoader):
        new_mrf = deepcopy(mrf)
        new_mrf.fit(dataloader)
        return new_mrf

    def rate_mrf(self, mrf, dataloader:torch.utils.data.DataLoader):
        score = torch.tensor(0.)
        for batch in dataloader:
            probs = mrf(batch).cpu()
            score += sum(probs)
        return score