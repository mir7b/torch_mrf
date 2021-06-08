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

    domain_lengths = torch.tensor([len(domain) for domain in domains])
    univserse_matrix_shape = (torch.prod(domain_lengths), torch.sum(domain_lengths))
    universe_matrix = torch.zeros(size=univserse_matrix_shape, dtype=torch.bool)
    
    for idx, world in tqdm.tqdm(enumerate(universe), total = len(universe_matrix), desc="Grounding Universe") if verbose else enumerate(universe):
        for jdx, feature in enumerate(world):
            universe_matrix[idx, torch.sum(domain_lengths[:jdx]):torch.sum(domain_lengths[:jdx+1])] = random_variables[jdx].encode(feature)
    
    return universe_matrix


def collapse_sideways(tensor):
    result = torch.ones(size=(tensor.shape[0],), dtype=torch.bool, device=tensor.get_device())
    for column in tensor.T:
        result *= column
    return result

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
    result = torch.ones(size=(w,b), dtype=torch.bool, device=tensor.get_device())
    for k in range(k):
        column = tensor[:,:,k]
        result *= column
        
    return result