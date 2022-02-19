import torch
import torch.nn as nn
from typing import List
import torch_random_variable.torch_random_variable as trv
import plotly.graph_objects as go
import tqdm
import torch.distributions as distributions



class GaussFactor(nn.Module):
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
                 max_parallel_worlds:int = pow(2,20), fill_value = 1., verbosity:int=1):
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
    
        super(GaussFactor, self).__init__()
        
        self.random_variables:List[trv.RandomVariable] = random_variables
        self.verbosity:int = verbosity
        self.device:str or int = device
        self.max_parallel_worlds:int = max_parallel_worlds
        
        self.discrete_random_variables = [rv for rv in self.random_variables if not isinstance(rv,trv.NumericRandomVariable)]
        self.continuous_random_variables = [rv for rv in self.random_variables if isinstance(rv,trv.NumericRandomVariable)]
        
        if len(self.discrete_random_variables) > 0:
            self.weights = nn.parameter.Parameter(data=torch.full(size=[var.domain_length for var in self.discrete_random_variables], 
                        fill_value=fill_value, dtype=torch.double, device=self.device))
        else:
            self.weights = torch.ones((1,))
            
        self.means = torch.zeros((*self.weights.shape, len(self.continuous_random_variables)))
        self.covariances = torch.zeros((*self.weights.shape, len(self.continuous_random_variables), len(self.continuous_random_variables)))
        self.determinants = nn.parameter.Parameter(data=torch.full(size=[var.domain_length for var in self.discrete_random_variables], 
                        fill_value=1., dtype=torch.double, device=self.device))
        self.inverses = torch.zeros((*self.weights.shape, len(self.continuous_random_variables), len(self.continuous_random_variables)))
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """ Takes a tensor with shape (samples, len(self.random_variables)) and returns the potential of each sample.

        Args:
            x (torch.Tensor): the samples. Must be indexable (long, binary, etc.)
        Returns:
            torch.Tensor: the potential
        """
        discrete_indices = self.discrete_rows()
        continuous_indices = self.continuous_rows()
        discrete_values = x[:, discrete_indices]
        discrete_probs = self.weights[discrete_values.chunk(len(self.discrete_random_variables),-1)].squeeze(-1)

    
    def discrete_rows(self) -> torch.Tensor:
        """Get the row indices of the random_variables in the universe.

        Returns:
            rows (torch.Tenosr<torch.long>): A tensor that contains the slices of the variables.
        """
        rows = []
        
        for idx, random_variable in enumerate(self.random_variables):
            if not isinstance(random_variable,trv.NumericRandomVariable):
                rows.append(idx)
                
        return torch.tensor(rows).long()
    
    
    def continuous_rows(self) -> torch.Tensor:
        """Get the row indices of the random_variables in the universe.

        Returns:
            rows (torch.Tenosr<torch.long>): A tensor that contains the slices of the variables.
        """
        rows = []
        
        for idx, random_variable in enumerate(self.random_variables):
            if isinstance(random_variable,trv.NumericRandomVariable):
                rows.append(idx)
                
        return torch.tensor(rows).long()

    
    def fit(self, data:torch.Tensor):
        """Calculates weights for this factor by infering the probability of each sample in the provided data.
        This is optimal w. r. t. the parameters as proven in TODO.

        Args:
            data (torch.Tensor): The data that is observed for this factor. data is a tensor of shape 
                                 (number of observations ,number of random_variables in this factor)) 
                                 and and indexable type like long or bool etc.
        """
        
        discrete_indices = self.discrete_rows()
        continuous_indices = self.continuous_rows()
        discrete_data = data[:,discrete_indices].long()
        print(discrete_data.shape)
        continuous_data = data[:,continuous_indices].float()
        
        with torch.no_grad():
            if len(self.discrete_random_variables) > 0:
                if self.verbosity <=1:
                    for state in torch.cartesian_prod(*[torch.arange(var.domain_length) for var in self.discrete_random_variables]):
                        samples = torch.all(discrete_data == state, dim=1).nonzero().squeeze()
                        prob = len(samples) / float(len(data))
                        self.weights[state] = prob
                        continuous_samples = continuous_data[samples]
                        self.means[state] = continuous_samples.mean(dim=0)
                        print(continuous_samples.shape)
                        covariance = continuous_samples.T.cov(correction=0)
                        print(covariance, continuous_samples.T.matmul(continuous_samples))
                        self.covariances[state] = covariance
                        self.inverses[state] = covariance.inverse()
                        self.determinants[state] = covariance.det()
                        
            else:
                self.means[0] = data.mean(dim=1)
                covariance = data.cov()
                self.covariances[0] = covariance
                self.inverses[0] = covariance.inverse()
                self.determinants[0] = covariance.det()
                
    
    def plot(self) -> go.Bar:
        """Visaulizes the potential of each world as bar trace.

        Returns:
            go.Bar: the bar trace
        """
        x = torch.cartesian_prod(*[torch.arange(var.domain_length) for var in self.random_variables]).to(self.device)
        y = self(x).detach().cpu()
        return go.Bar(x=[str(t.numpy()) for t in x.detach().cpu()], y=y, name="Distribution for %s" % self.random_variables)
    