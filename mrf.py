from numpy.core.fromnumeric import prod
import torch
import torch.nn as nn
import mrf_dataset
import os
import pracmln
import collections
import itertools
from itertools import chain, combinations
from torch_random_variable import RandomVariable

class MarkovRandomField(nn.Module):
    def __init__(self, random_variables, cliques):
        """Constructs a Markov Random Field from the nodes and edges."""

        #sort random variables
        self.random_variables = random_variables
        self.random_variables.sort(key=lambda x: x.name)

        for idx, clique in enumerate(cliques):
            for jdx, partner in enumerate(clique):
                if isinstance(partner, str):
                    cliques[idx][jdx] = RandomVariable(name=partner, domain = [var.domain for var  in self.random_variables if var.name ==partner][0])

        #sort the cliques
        [clique.sort(key=lambda x: x.name) for clique in cliques]
        self.cliques = cliques

        self._create_clique_universes()
        self._initialize_weights()

    def _create_clique_universes(self):
        """Create the universe for every clique as an indicator matrix.
        
        Every clique gets a universe matrix with the shape (num_worlds, num_features).
        This ensures that you can check for satasfied variables by multiplying.
        """
        self.clique_universes = dict()

        for clique in self.cliques:
            domains = [m.domain for m in clique]
            universe = itertools.product(*domains)

            domain_lengths = [len(domain) for domain in domains]
            univserse_matrix_shape = (prod(domain_lengths), sum(domain_lengths))
            universe_matrix = torch.zeros(size=univserse_matrix_shape, dtype=torch.bool)

            for idx, world in enumerate(universe):
                for jdx, feature in enumerate(world):
                    universe_matrix[idx, sum(domain_lengths[:jdx]):sum(domain_lengths[:jdx+1])] = clique[jdx].encode(feature)
            
            self.clique_universes[frozenset(clique)] = universe_matrix
            

    def _initialize_weights(self):
        """Initialize the weights for each world of each clique from a unifrom distribution with bounds of (0,1)."""
        self.clique_weights = dict()
        for clique, universe_matrix in self.clique_universes.items():
            self.clique_weights[clique] = torch.rand(size=(universe_matrix.shape[0],), 
                                                     dtype=torch.double, requires_grad=True)

    def _calc_z(self):
        self.Z=torch.tensor(1, dtype=torch.double)
        for _, weights in self.clique_weights.items():
            self.Z = self.Z * torch.prod(weights)
        print(self.Z)

    def forward(self, samples):
        self._calc_z()
        print(self.clique_universes)
        print(samples)
        

def main():

    path=os.path.join("..","pracmln","examples","alarm", "alarm.pracmln")
    mln_name="alarm-kreator.mln"
    db_name="query1.db"
    mln = pracmln.MLN.load(path + ":" + mln_name)
    database = pracmln.Database.load(mln, path + ":" + db_name)

    random_variables = [RandomVariable(name,domain) for name, domain in mln.domains.items() if name!="person"]

    mrf = MarkovRandomField(random_variables, [["domNeighborhood", "place"]])

    dataset = mrf_dataset.MRFDataset(mln=mln, database=database, random_variables=random_variables)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

    
    for batch in dataloader:
        mrf.forward(batch)

if __name__ == "__main__":
    main()