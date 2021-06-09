import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import mrf_dataset
import os
import pracmln
from torch_random_variable import RandomVariable
import mrf_utils
import tqdm
import frozenlist
import pytorch_lightning as pl


class MarkovRandomField(nn.Module):
    """
    Represents a Markov Random Field (MRF) from a set of random variables and cliques.
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
        """Constructs a Markov Random Field from the nodes and edges.
        
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

        for idx, clique in enumerate(cliques):
            for jdx, partner in enumerate(clique):
                if isinstance(partner, str):
                    cliques[idx][jdx] = RandomVariable(name=partner, domain = [var.domain for var  in self.random_variables if var.name ==partner][0])

        #sort the cliques
        self.cliques = [sorted(clique, key=lambda x: x.name) for clique in cliques]

        self._create_clique_universes()
        self._initialize_weights()
        self._calc_z()

    def _create_clique_universes(self):
        """Create the universe for every clique as an indicator matrix.
        
        Every clique gets a universe matrix with the shape (num_worlds, num_features).
        This ensures that you can check for satasfied variables by multiplying.
        """
        self.clique_universes = dict()
        
        #for every clique ground the universe
        for clique in tqdm.tqdm(self.cliques, desc="Grounding Cliques") if self.verbose else self.cliques:

            #cannot use frozen set here because iterating over frozen sets is somehow not deterministic
            key = frozenlist.FrozenList(clique)
            key.freeze()
            self.clique_universes[key] = mrf_utils.create_universe_matrix(clique).to(self.device)

        self.universe_matrix = mrf_utils.create_universe_matrix(self.random_variables)
        
    def _clip_weights(self):
        """Clip the weights of the Markov Random Field such that they are greater or equal to 0."""

        for clique, weights in self.clique_weights.items():
            self.clique_weights[clique].data = F.relu(weights)

    def _initialize_weights(self):
        """Initialize the weights for each world of each clique from a unifrom distribution with bounds of (0,1)."""
        self.clique_weights = nn.ParameterDict()
        for clique, universe_matrix in self.clique_universes.items():
            self.clique_weights[str(clique)] = nn.parameter.Parameter(data=torch.rand(size=(universe_matrix.shape[0],), 
                                                     dtype=torch.double, device=self.device))
            #self.register_parameter(str(clique), self.clique_weights[clique])


    def _get_rows_in_universe(self, random_variables):
        """Get the row indices of the random_variables in the universe.
        
        Args:
            random_variables (iterable<torch_random_variable.RandomVariable>): The random variables which row
            slices are desired

        Returns:
            rows (torch.Tenosr<torch.long>): A tensor that contains the slices of the variables.
        """

        world_domain_lengths = torch.tensor([variable.domain_length for variable in self.random_variables])
        
        local_domain_lengths = torch.tensor([variable.domain_length for variable in random_variables])

        rows = torch.zeros(size=(torch.sum(local_domain_lengths),), dtype=torch.long)
        row_idx = 0

        for idx, variable in enumerate(random_variables):
            pos = self.random_variables.index(variable)
            begin = torch.sum(world_domain_lengths[:pos])
            end = torch.sum(world_domain_lengths[:pos+1])
            for value in range(begin, end):
                rows[row_idx] = value
                row_idx += 1

        return rows

    def _calc_z(self):
        """Calculate the probability mass of this mrf with respect to the current weights.
        
        """
        self.Z = torch.tensor(1, dtype=torch.double, device=self.device)

        new_Z = torch.tensor(0, dtype=torch.double, device=self.device)
        num_batches = int(len(self.universe_matrix) / self.max_parallel_worlds) + 1
        
        for i in range(num_batches):
            #get max_parallele_worlds amount of worlds from the universe
            worlds = self.universe_matrix[i*self.max_parallel_worlds : (i+1) * self.max_parallel_worlds].to(self.device)

            #calc their probability mass
            probabilities = self(worlds)

            new_Z += torch.sum(probabilities)

        self.Z = new_Z


    def forward(self, samples):
        """Get the probabilities of a batch of worlds. The batch has to have the shape (num_samples, num_world_features).
        
        Args:
            samples (tensor<torch.bool>): The worlds which probabilities should be calculated

        Returns probabilities (tensor<double>): A tensor with the probability of each sample.
        """
        b, k = samples.shape
        world_probability_masses = torch.ones(size=(b,), dtype=torch.double, device=self.device)

        for clique, clique_universe in self.clique_universes.items():
                rows = self._get_rows_in_universe(clique)

                clique_features = samples[:,rows]

                fitting_worlds = torch.repeat_interleave(clique_universe.unsqueeze(1), b, 1) == clique_features
                collapsed = mrf_utils.batch_collapse_sideways(fitting_worlds)

                clique_weights = collapsed * torch.repeat_interleave(self.clique_weights[str(clique)].unsqueeze(1), b, 1)

                clique_weights = torch.sum(clique_weights, dim=-2)
                
                world_probability_mass = world_probability_masses * clique_weights
        
        return world_probability_mass / self.Z


def main():

    path=os.path.join("..","pracmln","examples","alarm", "alarm.pracmln")
    mln_name="alarm-kreator.mln"
    db_name="query1.db"
    mln = pracmln.MLN.load(path + ":" + mln_name)
    database = pracmln.Database.load(mln, path + ":" + db_name)

    random_variables = [RandomVariable(name,domain) for name, domain in mln.domains.items() if name!="person"]

    mrf = MarkovRandomField(random_variables, [["domNeighborhood"],["place"]], device="cuda")

    dataset = mrf_dataset.MRFDataset(mln=mln, database=database, random_variables=random_variables)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(mrf.parameters(), lr=0.1)
    

    pbar = tqdm.tqdm(range(1000), desc="Training mrf")
    for _ in pbar:
        for batch in dataloader:
            preds = mrf.forward(batch.to(mrf.device))

            loss = criterion(preds, torch.ones(size=preds.shape, dtype = torch.double, device = preds.device))
            loss.backward(retain_graph=True)
            
            optimizer.step()
            optimizer.zero_grad()
            
            mrf._clip_weights()
            mrf._calc_z()

            pbar.set_postfix(train_loss=loss.item())
    
    print(preds)
    print(list(mrf.parameters()))  
    
    

if __name__ == "__main__":
    main()