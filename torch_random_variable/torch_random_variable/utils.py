import torch

def binary(x, bits):
    """Convert tensor of longs to binary tensors of *bits* length.
    
    Args:
        x (torch.Tensor<torch.long>): The tensor that shall be binarized
        bits (int): The length of each binary tensor
    
    Returns:
        The binary representation of the input
    """
    mask = 2**torch.arange(bits).to(x.device, torch.long)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).bool()

def decimal(b, bits):
    """Convert tensor of booleans to decimal tensor.
    
    Args:
        x (torch.Tensor<torch.bool>): The tensor that shall be converted to decimal
        bits (int): The length of each binary tensor
    
    Returns:
        The decimal representation of the input
    """
    mask = 2 ** torch.arange(bits).to(b.device, torch.long)
    return torch.sum(mask * b, -1)