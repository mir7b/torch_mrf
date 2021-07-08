"""This file describes vecotrized Markov Random Fields for the GPU."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_mrf import mrf_utils
import tqdm
import math
import frozenlist



class MarkovRandomField(nn.Module):
    """Represents a Markov Random Field (MRF) from a set of random variables and cliques.
    
    The MRF is highly vectorized and can be used for partial queries.

    Attributes:
        random_variables (iterable<torch_random_variable.RandomVariable>): The random variables that are represented
            in this Markov Random Field
        cliques (iterable<iterable<torch_random_variable.RandomVariable>>): The connectivity of the random variables
            as a list of lists where the inner lists represent all members of a clique. Can also be a list of lists of
            strings, where the strings are the variable names of the clique members.
        device (str): The device where the Markov Random Field will perform most of its calculations
        max_parallel_worlds (int): The maximum number of worlds that are evaluated at once on the graphics card.
            This parameter can produce an Cuda Out of Memory Error when picked too large and slow down speed when picked too small.
        verbose (bool): Whether to show progression bars or not
        Z (torch.Tensor<torch.double>): The overall probability mass.
        universe_matrix (torch.Tensor<torch.bool>): The whole universe that can be created from all random variables of the MRF.
        clique_universes (dict<frozenlist<torch_random_variable.RandomVariable>, torch.Tensor>): The universes
            that get covered by each clique.
        clique_weights (dict<str, torch.nn.parameter.Parameter>): A dict that maps the cliques to the weights for 
            each clique which will be optimized.
        
    """

    def __init__(self, random_variables, cliques, device="cuda", max_parallel_worlds = pow(2,20),verbose=1):
        """Construct a Markov Random Field from the nodes and edges.

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
        super(MarkovRandomField, self).__init__()

        #sort random variables
        self.random_variables = random_variables
        self.random_variables.sort(key=lambda x: x.name)
        self.verbose = verbose
        self.device = device
        self.max_parallel_worlds = max_parallel_worlds

        #parse clique members to variable if they arent already variables
        for idx, clique in enumerate(cliques):
            for jdx, partner in enumerate(clique):
                if isinstance(partner, str):
                    cliques[idx][jdx], = [var for var in self.random_variables if var.name==partner]

        #sort the cliques and convert them to hashable datatypes
        self.cliques = [sorted(clique, key=lambda x: x.name) for clique in cliques]
        self.cliques = [frozenlist.FrozenList(clique) for clique in self.cliques]
        for clique in self.cliques:
            clique.freeze()


        self._initialize_weights()
        self._init_rows_in_univserse()

        #self.calc_z()
        #by intializing all weights with 1 we can skip the first Z calculation and exploit 1 being the netraul
        #multiplication element
        self.Z = torch.prod(torch.tensor([var.domain_length for var in self.random_variables]))

    def _init_rows_in_univserse(self):
        """Initialize the indices of the rows of every clique and random variable in the universe matrix."""
        self.random_variable_row_in_univserse = dict()
        self.clique_rows_in_univserse = dict()
        
        for random_variable in self.random_variables:
            self.random_variable_row_in_univserse[random_variable] = self._get_rows_in_universe([random_variable])
        
        for clique in self.cliques:
            self.clique_rows_in_univserse[clique] = self._get_rows_in_universe(clique)
        

    def clip_weights(self):
        """Clip the weights of the Markov Random Field such that they are greater or equal to 0."""
        for clique, weights in self.clique_weights.items():
            self.clique_weights[clique].data = F.relu(weights)


    def _initialize_weights(self, fill_value=1):
        """Initialize the weights for each world of each clique from a unifrom distribution with bounds of (0,1)."""
        self.clique_weights = nn.ParameterDict()
        for clique in self.cliques:
            num_worlds = torch.prod(torch.tensor([var.domain_length for var in clique]))
            #have to use the string representation here because parameter dicts only allow strings as keys
            self.clique_weights[str(clique)] = nn.parameter.Parameter(data=torch.full(size=(num_worlds,), 
                                            fill_value=fill_value, dtype=torch.double, device=self.device))
            

    def _get_rows_in_universe(self, random_variables):
        """Get the row indices of the random_variables in the universe.
        
        Args:
            random_variables (iterable<torch_random_variable.RandomVariable>): The random variables which row
            slices are desired

        Returns:
            rows (torch.Tenosr<torch.long>): A tensor that contains the slices of the variables.
        """
        world_encoding_lengths = torch.tensor([variable.encoding_length for variable in self.random_variables])
        
        local_encoding_lengths = torch.tensor([variable.encoding_length for variable in random_variables])

        rows = torch.zeros(size=(torch.sum(local_encoding_lengths),), dtype=torch.long)
        row_idx = 0

        for variable in random_variables:
            pos = self.random_variables.index(variable)
            begin = torch.sum(world_encoding_lengths[:pos])
            end = torch.sum(world_encoding_lengths[:pos+1])
            for value in range(begin, end):
                rows[row_idx] = value
                row_idx += 1

        return rows

    def _get_rows_in_clique(self, clique, random_variables):
        """Get the row indices of the random_variables in the clique universe.
        
        Args:
            random_variables (iterable<torch_random_variable.RandomVariable>): The random variables which row
            slices are desired

        Returns:
            rows (torch.Tenosr<torch.long>): A tensor that contains the slices of the variables.
        """
        world_encoding_lengths = torch.tensor([variable.encoding_length for variable in self.random_variables])
        
        local_encoding_lengths = torch.tensor([variable.encoding_length for variable in random_variables])

        rows = torch.zeros(size=(torch.sum(local_encoding_lengths),), dtype=torch.long)
        row_idx = 0

        for variable in random_variables:
            pos = clique.index(variable)
            begin = torch.sum(world_encoding_lengths[:pos])
            end = torch.sum(world_encoding_lengths[:pos+1])
            for value in range(begin, end):
                rows[row_idx] = value
                row_idx += 1

        return rows

    def calc_z(self):
        """Calculate the probability mass of this mrf with respect to the current weights."""
        
        #reset Z to 1 for the forward calculations
        self.Z = torch.tensor(1, dtype=torch.double, device=self.device, requires_grad=False)

        #initialize new Z
        new_Z = torch.tensor(0, dtype=torch.double, device=self.device)

        max_worlds = math.floor(math.sqrt(self.max_parallel_worlds))

        #iterate over world batches
        for world_batch in mrf_utils.iter_universe_batches(self.random_variables, max_worlds=max_worlds, 
                                                           verbose=self.verbose > 0):
            
            #calc their probability mass
            probabilities = self(world_batch.to(self.device))

            #add up the new overall probability mass
            new_Z += torch.sum(probabilities)
            
        #set it as class variable
        self.Z = new_Z
        

    def predict(self, samples):
        """Return the probability of each sample in the input. The samples can be partial worlds.
        
        Args:
            samples (iterable<dict<torch_random_variable.RandomVariable, str>>): The batch of (partial) worlds
                Can also be a dict that mapes the variable name to its value.

        Returns:
            probabilities (torch.Tensor<torch.double>): Tensor of probabilites with same length as samples.

        """
        #vector to store all probabilites
        probabilities = torch.zeros((len(samples),), dtype=torch.double, device=self.device)

        for idx, sample in enumerate(samples):
            #get the features
            rows = None
            values = None
            for variable, value in sample.items():
                if isinstance(variable,str):
                    variable, = [var for var in self.random_variables if var.name==variable]

                if rows is None:
                    rows = self.random_variable_row_in_univserse[variable]
                else:
                    rows = torch.cat((rows, self.random_variable_row_in_univserse[variable]))

                #if it is a binary variable dont use True or False, use [True] or [False] instead
                encoded = variable.encode(value)
                if len(encoded.shape) == 0:
                    encoded = encoded.unsqueeze(0)

                if values is None:
                    values = encoded
                else:
                    values = torch.cat((values, encoded))
            
            for world_batch in mrf_utils.iter_universe_batches(self.random_variables, self.max_parallel_worlds):
                worlds = torch.where(world_batch[:,rows] == values, True, False).bool().to(self.device)
                satasfied_worlds = mrf_utils.batch_collapse_sideways(worlds.unsqueeze(1)).squeeze()
 
                satasfied_worlds_indices = satasfied_worlds * torch.tensor(range(1,len(world_batch)+1), dtype=torch.long, device=self.device)
                satasfied_worlds_indices = satasfied_worlds_indices[satasfied_worlds_indices.nonzero(as_tuple=True)] -1
                probability = self(world_batch[satasfied_worlds_indices].to(self.device)).sum()
                probabilities[idx] += probability
        
        return probabilities

    def _forward_world_batch(self, clique_features, world_batch):
        b = len(clique_features)
        #move world batch to own device
        world_batch = world_batch.to(self.device).detach()
        #collapse the worlds where the clique feature holds
        fitting_worlds = torch.repeat_interleave(world_batch.unsqueeze(1), b, 1).detach() == clique_features

        collapsed = mrf_utils.batch_collapse_sideways(fitting_worlds)
        return collapsed

    def _weight_collapsed_batch(self, weights, collapsed):
        _,b = collapsed.shape
        #get the weight of each holding clique
        clique_weights = collapsed * torch.repeat_interleave(weights.unsqueeze(1), b, 1)
        return clique_weights

    def forward(self, samples):
        """Get the probabilities of a batch of worlds. The batch has to have the shape (num_samples, num_world_features).
        
        Args:
            samples (tensor<torch.bool>): The worlds which probabilities should be calculated

        Returns probabilities (tensor<double>): A tensor with the probability of each sample.
        """

        #get batch and world dimension
        b, k = samples.shape

        #construct result probability masses as neutral element of multiplication
        world_probability_masses = torch.ones(size=(b,), dtype=torch.double, device=self.device)

        #calculate the amount of worlds that can be processed w. r. t. the batch size
        #effective_parallel_worlds =  max(1, math.floor(self.max_parallel_worlds/b))

        #for every clique
        for clique in self.cliques:
                #get the rows of the clique in the universe
                rows = self.clique_rows_in_univserse[clique]
                
                #get the clique features of each world
                clique_features = samples[:,rows].detach()
                
                world_begin = 0
                #for every batch of worlds
                for world_batch in mrf_utils.iter_universe_batches(list(clique),self.max_parallel_worlds,
                                                                   verbose = self.verbose > 1):
                    
                    collapsed = self._forward_world_batch(clique_features,world_batch)

                    #get the amount of worlds in the world batch
                    w,_ = world_batch.shape

                    #track beginning of world and ending of world to use the right weights.
                    world_end = world_begin + w
                    
                    #get the clique weights for the current worlds
                    clique_weights = self.clique_weights[str(clique)][world_begin:world_end]

                    #get the weight of each holding clique
                    clique_weights = self._weight_collapsed_batch(clique_weights, collapsed)
                    
                    #clique_weights = collapsed * torch.repeat_interleave(clique_weights.unsqueeze(1), b, 1)

                    #sum up the weights
                    clique_weights = torch.sum(clique_weights, dim=-2)

                    #multiply with the weight of each previous clique
                    world_probability_mass = world_probability_masses * clique_weights
                    world_begin = world_end
                    
        #scale by the overall probability mass
        return world_probability_mass / self.Z
        
    def fit(self, dataloader):
        """Fit the model to the conjunctive probability distribution of the data without having to rely 
           on gradient descent (which has to recalculate Z every once in a while.)"""

        dataset_length = len(dataloader.dataset)
        if self.verbose > 0:
            pbar = tqdm.tqdm(dataloader, desc="Fitting MRF", 
                             total=math.ceil(dataset_length / dataloader.batch_size))

        self._initialize_weights(0)

        #for every batch provided by the dataloader
        for batch in pbar if self.verbose > 0 else dataloader:

            #for every clique
            for clique in self.cliques:
                #get the rows of the clique in the universe
                rows = self.clique_rows_in_univserse[clique]
                
                #get the clique features of each world
                clique_features = batch[:,rows].detach().to(self.device)

                world_begin = 0

                #for every batch of worlds
                for world_batch in mrf_utils.iter_universe_batches(list(clique),self.max_parallel_worlds,
                                                                   verbose = self.verbose > 1):
                    #get the amount of worlds in the world batch
                    w,_ = world_batch.shape

                    #track beginning of world and ending of world to use the right weights.
                    world_end = world_begin + w

                    collapsed = self._forward_world_batch(clique_features,world_batch.to(self.device))
                    summed = torch.sum(collapsed,dim=1)
                    with torch.no_grad():
                        self.clique_weights[str(clique)][world_begin:world_end] = self.clique_weights[str(clique)][world_begin:world_end] + (summed/ torch.full(summed.shape, dataset_length).to(self.device)).to(self.device)

                    world_begin = world_end