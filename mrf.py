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

        #sort the cliques
        [clique.sort(key=lambda x: x.name) for clique in cliques]
        self.cliques = cliques

        # self._create_clique_universes()
        # self._initialize_weights()

    def _create_clique_universes(self):

        self.clique_horizons = dict()
        self.clique_universes = dict()

        #this part creates the horizon of all cliques in a readable way.
        #for all cliques
        for clique in self.cliques:

            #create a horizon
            horizon = []

            #for every random variable that is part of this clique
            for domain in clique:
                grounded_domain = []

                #ground the variable by all possible values
                for value in self.domains[domain]:
                    grounded_domain.append((domain,value))

                #add the grounding to the universe
                horizon.append(grounded_domain)

            #save the universe of this clique for runtime reduction
            self.clique_horizons[frozenset(clique)] = horizon

        #this part creates the universe of all cliques as indicator funtion way (which is hard to interpret)
        for clique, horionz in self.clique_horizons.items():

            #calculate the shape of the universe matrix which is (num_worlds x num_features)
            domain_lengths = torch.tensor([len(h) for h in horionz])
            num_features = torch.sum(domain_lengths)
            num_worlds = torch.prod(domain_lengths)

            #create matrix to hold universes
            universe_matrix = torch.zeros(size=(num_worlds,num_features), dtype=torch.bool)

            #create all possible universe
            universe = list(itertools.product(*horizon))

            #extract the features from every world and write them into the universe matrix
            for idx, world in enumerate(universe):
                for jdx, feature in enumerate([item for sublist in horizon for item in sublist]):
                    universe_matrix[idx,jdx] = feature in world
            
            self.clique_universes[clique] = universe_matrix

    def _initialize_weights(self):
        """Initialize the weights for each world of each clique"""
        self.clique_weights = dict()
        for clique, universe_matrix in self.clique_universes.items():
            self.clique_weights[clique] = torch.rand(size=(universe_matrix.shape[0],), 
                                                     dtype=torch.double, requires_grad=True)

    def forward(self, samples):
        print(samples)
        

def main():

    path=os.path.join("..","pracmln","examples","alarm", "alarm.pracmln")
    mln_name="alarm-kreator.mln"
    db_name="query1.db"
    mln = pracmln.MLN.load(path + ":" + mln_name)
    database = pracmln.Database.load(mln, path + ":" + db_name)

    dataset = mrf_dataset.MRFDataset(mln=mln, database=database)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
    mrf = MarkovRandomField(mln.domains, [["domNeighborhood", "place"]])
    
    for batch in dataloader:
        mrf.forward(batch)

if __name__ == "__main__":
    main()