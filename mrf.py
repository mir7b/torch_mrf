import torch
import torch.nn as nn
import mrf_dataset
import os
import pracmln
from torch_random_variable import RandomVariable
import mrf_utils
import tqdm
import frozenlist

class MarkovRandomField(nn.Module):
    def __init__(self, random_variables, cliques, device="cuda", verbose=True):
        """Constructs a Markov Random Field from the nodes and edges."""

        #sort random variables
        self.random_variables = random_variables
        self.random_variables.sort(key=lambda x: x.name)
        self.verbose = verbose
        self.device = device

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
        


    def _initialize_weights(self):
        """Initialize the weights for each world of each clique from a unifrom distribution with bounds of (0,1)."""
        self.clique_weights = dict()
        for clique, universe_matrix in self.clique_universes.items():
            self.clique_weights[clique] = torch.rand(size=(universe_matrix.shape[0],), 
                                                     dtype=torch.double, requires_grad=True, device=self.device)


    def _get_rows_in_universe(self, random_variables):
        """Get the row slices of the random_variables in the universe.
        
        Args:
            random_variables (iterable<torch_random_variable.RandomVariable>): The random variables which row
            slices are desired

        Returns:
            rows (torch.Tenosr<torch.long>): A tensor that contains the slices of the variables.
        """
        domains = [variable.domain for variable in self.random_variables]
        domain_lengths = torch.tensor([len(domain) for domain in domains])

        rows = torch.zeros(size=(torch.sum(domain_lengths),), dtype=torch.long)
        row_idx = 0

        for idx, variable in enumerate(random_variables):
            pos = self.random_variables.index(variable)
            begin = torch.sum(domain_lengths[:pos])
            end = torch.sum(domain_lengths[:pos+1])
            for value in range(begin, end):
                rows[row_idx] = value
                row_idx += 1

        return rows

    def _calc_z(self):
        """Calculate the probability mass of this mrf with respect to the current weights.
        
        """

        #reset Z to 0
        self.Z=torch.tensor(0, dtype=torch.double, device=self.device)
        
        #for every world in the universe
        for world in tqdm.tqdm(self.universe_matrix, desc="Calculate overall probability mass") if self.verbose else self.universe_matrix:
            
            #initialize the probability mass of this world as 0
            world_probability_mass = torch.tensor(1, dtype=torch.double, device=self.device)

            #for every clique calc the weight of this world
            for clique, clique_universe in self.clique_universes.items():

                #get feature row indices of this clique with respect to the universe
                rows = self._get_rows_in_universe(clique)
                
                #get clique features
                clique_features = world[rows].to(self.device)

                #check which clique worlds are inline with the current world
                fitting_worlds = clique_universe == clique_features
                collapsed = mrf_utils.collapse_sideways(fitting_worlds)
                
                #calculate weights for the worlds that are satasfied
                clique_weight = torch.sum(collapsed * self.clique_weights[clique])
                
                #multiply the world probability mass
                world_probability_mass = world_probability_mass * clique_weight

            #sum the world probability masses up
            self.Z = self.Z + world_probability_mass
        
        

    def forward(self, samples):
        b, k = samples.shape
        world_probability_masses = torch.ones(size=(b,), dtype=torch.double, device=self.device)

        for clique, clique_universe in self.clique_universes.items():
                rows = self._get_rows_in_universe(clique)
                
                clique_features = samples[:,rows]
                fitting_worlds = torch.repeat_interleave(clique_universe.unsqueeze(1), b, 1) == clique_features
                collapsed = mrf_utils.batch_collapse_sideways(fitting_worlds)

                clique_weights = collapsed * torch.repeat_interleave(self.clique_weights[clique].unsqueeze(1), b, 1)

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

    mrf = MarkovRandomField(random_variables, [["domNeighborhood", "place"]])

    dataset = mrf_dataset.MRFDataset(mln=mln, database=database, random_variables=random_variables)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

    
    for batch in dataloader:
        preds = mrf.forward(batch.to(mrf.device))
        print(preds)
if __name__ == "__main__":
    main()