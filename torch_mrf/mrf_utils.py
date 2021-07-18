"""Utility functions for markov random fields that maybe needed outside of mrfs."""

import torch
import itertools
import tqdm

def create_universe_matrix(random_variables, verbose=False):
    """Calculate the universe of a set of random variables.

    Args:
        random_variables (iterable<torch_random_variable.RandomVariable>): The iterable of random variables 
        verbose (bool): Rather to print a tqdm loading bar or not.
    Returns:
        universe_matrix (torch.Tensor<torch.bool>): A matrix that contains all valid combinations of the input

    """
    #create all combinations of worlds
    domains = [torch.tensor(range(var.domain_length)) for var in random_variables]
    universe = torch.cartesian_prod(*domains)

    #calc shape for universe matrix and create it
    domain_lengths = torch.tensor([var.domain_length for var in random_variables])
    encoding_lengths = torch.tensor([var.encoding_length for var in random_variables])
    univserse_matrix_shape = (torch.prod(domain_lengths), torch.sum(encoding_lengths))
    universe_matrix = torch.zeros(size=univserse_matrix_shape, dtype=torch.bool)

    for idx, column in enumerate(tqdm.tqdm(universe.T, desc="Grounding Universe")) if verbose else enumerate(universe.T):
        universe_matrix[:, torch.sum(encoding_lengths[:idx]):torch.sum(encoding_lengths[:idx+1])] = binary(column, random_variables[idx].encoding_length)

    return universe_matrix

def iter_universe_batches(random_variables, max_worlds=pow(2,20), verbose=False):

    random_variables = random_variables.copy()
    domain_lengths = torch.tensor([var.domain_length for var in random_variables])
    encoding_lengths = torch.tensor([var.encoding_length for var in random_variables])
    trimmed_domain_lengths = domain_lengths.clone()

    #trimm the domain lengths of the variables to reduce the size of the generated mini-universes
    while torch.prod(trimmed_domain_lengths) > max_worlds:
        for idx, domain_length in enumerate(trimmed_domain_lengths):
            if domain_length > 1:
                trimmed_domain_lengths[idx] -= 1
                break

    chunked_domains = [list(chunks(torch.tensor(range(var.domain_length)), trimmed_domain_lengths[idx]))for idx, var in enumerate(random_variables)]

    if verbose:
        pbar =  tqdm.tqdm(itertools.product(*chunked_domains), desc="Iterating Worlds",
                          total=torch.prod(torch.tensor([len(dom) for dom in chunked_domains])).item())
    for chunked_domain in pbar if verbose else itertools.product(*chunked_domains):
        #create all worlds as integer encoding
        universe_batch = torch.cartesian_prod(*chunked_domain)

        #convert that to binary encoding
        binary_universe_batch = torch.zeros(size=(len(universe_batch), torch.sum(encoding_lengths)), dtype=torch.bool)
        for idx, column in enumerate(universe_batch.T):
            binary_universe_batch[:, torch.sum(encoding_lengths[:idx]):torch.sum(encoding_lengths[:idx+1])] = binary(column, random_variables[idx].encoding_length)
        
        #yield the binary universe batch
        yield binary_universe_batch.detach()
        
    


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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]