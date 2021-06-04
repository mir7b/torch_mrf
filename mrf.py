import torch
import torch.nn as nn
import mrf_dataset
import os
import pracmln
import collections
import itertools
from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class MarkovRandomField(nn.Module):
    def __init__(self, mln, cliques):
        """Constructs a Markov Random Field from the nodes and edges."""
        self.mln = mln
        self.cliques = cliques

        self._create_universe()

    def _create_universe(self):

        #sort domains and domain values for determinism reasons
        self.sorted_domains = dict()
        for key, value in sorted(self.mln.domains.items()):
            self.sorted_domains[key] = sorted(value)
        
        universe = torch.zeros(size=())
        for domain, values in self.sorted_domains.items():
            ps = list(powerset(values))        
            print(ps)

def main():

    path=os.path.join("..","pracmln","examples","alarm", "alarm.pracmln")
    mln_name="alarm-kreator.mln"
    db_name="query1.db"
    
    mln = pracmln.MLN.load(path + ":" + mln_name)
    database = pracmln.Database.load(mln, path + ":" + db_name)

    dataset = mrf_dataset.MRFDataset(mln=mln, database=database)

    dataloader = torch.utils.data.DataLoader(dataset)

    mrf = MarkovRandomField(mln, [["person", "domNeighborhood", "place"]])


if __name__ == "__main__":
    main()