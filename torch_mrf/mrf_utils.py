"""Utility functions for markov random fields that maybe needed outside of mrfs."""

import torch
import itertools
import tqdm

def create_universe_matrix(random_variables, verbose=True):
    """Calculate the universe of a set of random variables.

    Args:
        random_variables (iterable<torch_random_variable.RandomVariable>): The iterable of random variables 
    
    Returns:
        universe_matrix (torch.Tensor<torch.bool>): A matrix that contains all valid combinations of the input

    """
    domains = [variable.domain for variable in random_variables]
    universe = itertools.product(*domains)
    
    domain_lengths = torch.tensor([var.domain_length for var in random_variables])
    encoding_lengths = torch.tensor([var.encoding_length for var in random_variables])

    univserse_matrix_shape = (torch.prod(domain_lengths), torch.sum(encoding_lengths))
    universe_matrix = torch.zeros(size=univserse_matrix_shape, dtype=torch.bool)
    
    for idx, world in tqdm.tqdm(enumerate(universe), total = len(universe_matrix), desc="Grounding Universe") if verbose else enumerate(universe):
        for jdx, feature in enumerate(world):
            universe_matrix[idx, torch.sum(encoding_lengths[:jdx]):torch.sum(encoding_lengths[:jdx+1])] = random_variables[jdx].encode(feature)
    
    return universe_matrix

def batch_collapse_sideways(tensor):
    """Let a tensor with shape (w, b, k) collaps to shape (w, b) by multiplying the k dimension into each other.
    
    In the case of TorchMRFs w stands for the number of worlds, b for the number of queries and k for the product of
    the domains of the world (the features). 

    Args:
        tensor (torch.Tensor): The tensor that will be collapsed of shape (w,b,k)
    
    Returns:
        tensor (torch.Tensor): The collapsed tensor of shape (w,b)
    """
    #num worlds, num batches, dim per world is the input tensors shape
    w, b, k = tensor.shape
    result = torch.ones(size=(w,b), dtype=torch.bool, device=tensor.device)
    for k in range(k):
        column = tensor[:,:,k]
        result *= column
        
    return result

def binary(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()