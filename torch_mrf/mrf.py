"""This file describes vecotrized Markov Random Fields for the GPU."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import mrf_utils
import tqdm
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

    def __init__(self, random_variables, cliques, device="cuda", max_parallel_worlds = 1024,verbose=True):
        """Construct a Markov Random Field from the nodes and edges.

        Args:
            random_variables (iterable<torch_random_variable.RandomVariable>): The random variables that are represented
                in this Markov Random Field
            cliques (iterable<iterable<torch_random_variable.RandomVariable>>): The connectivity of the random variables
                as a list of lists where the inner lists represent all members of a clique
            device (str): The device where the Markov Random Field will perform most of its calculations
            max_parallel_worlds (int): The maximum number of worlds that are evaluated at once on the graphics card.
                This parameter can produce an Cuda Out of Memory Error when picked too large and slow down speed when picked too small.
            verbose (bool): Whether to show progression bars or not

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

        self._create_clique_universes()
        self._initialize_weights()
        self._init_rows_in_univserse()
        self.calc_z()


    def _init_rows_in_univserse(self):
        """Initialize the indices of the rows of every clique and random variable in the universe matrix."""
        self.random_variable_row_in_univserse = dict()
        self.clique_rows_in_univserse = dict()
        
        for random_variable in self.random_variables:
            self.random_variable_row_in_univserse[random_variable] = self._get_rows_in_universe([random_variable])
        
        for clique in self.cliques:
            self.clique_rows_in_univserse[clique] = self._get_rows_in_universe(clique)


    def _create_clique_universes(self):
        """Create the universe for every clique as an indicator matrix.
        
        Every clique gets a universe matrix with the shape (num_worlds, num_features).
        This ensures that you can check for satasfied variables by multiplying.
        """
        self.clique_universes = dict()
        
        #for every clique ground the universe
        for clique in tqdm.tqdm(self.cliques, desc="Grounding Cliques") if self.verbose else self.cliques:

            self.clique_universes[clique] = mrf_utils.create_universe_matrix(clique, verbose=False).to(self.device)

        self.universe_matrix = mrf_utils.create_universe_matrix(self.random_variables, verbose=self.verbose)
        

    def clip_weights(self):
        """Clip the weights of the Markov Random Field such that they are greater or equal to 0."""
        for clique, weights in self.clique_weights.items():
            self.clique_weights[clique].data = F.relu(weights)


    def _initialize_weights(self):
        """Initialize the weights for each world of each clique from a unifrom distribution with bounds of (0,1)."""
        self.clique_weights = nn.ParameterDict()
        for clique, universe_matrix in self.clique_universes.items():
            #have to use the string representation here because parameter dicts only allow strings as keys
            self.clique_weights[str(clique)] = nn.parameter.Parameter(data=torch.rand(size=(universe_matrix.shape[0],), 
                                                     dtype=torch.double, device=self.device))


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

        for idx, variable in enumerate(random_variables):
            pos = self.random_variables.index(variable)
            begin = torch.sum(world_encoding_lengths[:pos])
            end = torch.sum(world_encoding_lengths[:pos+1])
            for value in range(begin, end):
                rows[row_idx] = value
                row_idx += 1

        return rows

    def calc_z(self):
        """Calculate the probability mass of this mrf with respect to the current weights."""
        #reset Z to 1 for the forward calculations
        self.Z = torch.tensor(1, dtype=torch.double, device=self.device)

        #initialize new Z
        new_Z = torch.tensor(0, dtype=torch.double, device=self.device)
        
        #get batches of universe
        num_batches = int(len(self.universe_matrix) / self.max_parallel_worlds) + 1
        
        for i in range(num_batches):
            #get max_parallele_worlds amount of worlds from the universe
            worlds = self.universe_matrix[i*self.max_parallel_worlds : (i+1) * self.max_parallel_worlds].to(self.device)

            #calc their probability mass
            probabilities = self(worlds)

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
            
            worlds = torch.where(self.universe_matrix[:,rows] == values, 1, 0).bool().to(self.device)
            satasfied_worlds = mrf_utils.batch_collapse_sideways(worlds.unsqueeze(1)).squeeze()
 
            satasfied_worlds_indices = satasfied_worlds * torch.tensor(range(1,len(self.universe_matrix)+1), dtype=torch.long, device=self.device)
            
            satasfied_worlds_indices = satasfied_worlds_indices[satasfied_worlds_indices.nonzero(as_tuple=True)] -1
            probability = torch.sum(self(self.universe_matrix[satasfied_worlds_indices].to(self.device)))
            probabilities[idx] = probability
        
        return probabilities


    def forward(self, samples):
        """Get the probabilities of a batch of worlds. The batch has to have the shape (num_samples, num_world_features).
        
        Args:
            samples (tensor<torch.bool>): The worlds which probabilities should be calculated

        Returns probabilities (tensor<double>): A tensor with the probability of each sample.
        """
        #get batch and world dimension
        b, k = samples.shape

        #construct result probability masses
        world_probability_masses = torch.ones(size=(b,), dtype=torch.double, device=self.device)

        #for every clique
        for clique, clique_universe in self.clique_universes.items():

                #get the rows of the clique in the universe
                rows = self.clique_rows_in_univserse[clique]

                #get the clique features of each world
                clique_features = samples[:,rows]

                #collapse the worlds where the clique feature holds
                fitting_worlds = torch.repeat_interleave(clique_universe.unsqueeze(1), b, 1) == clique_features
                collapsed = mrf_utils.batch_collapse_sideways(fitting_worlds)

                #get the weight of each holding clique
                clique_weights = collapsed * torch.repeat_interleave(self.clique_weights[str(clique)].unsqueeze(1), b, 1)

                #sum up the weights
                clique_weights = torch.sum(clique_weights, dim=-2)
                
                #multiply with the weight of each previous clique
                world_probability_mass = world_probability_masses * clique_weights
        
        #scale by the overall probability mass
        return world_probability_mass / self.Z