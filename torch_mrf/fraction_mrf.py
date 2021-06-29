from torch_mrf import mrf
from torch_fraction import fraction
import tqdm
import math
from torch_mrf import mrf_utils
import torch

class FractionMRF(mrf.MarkovRandomField):
    def __init__(self, random_variables, cliques, device="cuda", max_parallel_worlds = pow(2,20),verbose=1):
        super(FractionMRF, self).__init__(random_variables, cliques, device, max_parallel_worlds, verbose)
        self.Z = 1

    def _initialize_weights(self, denominator=1):
        self.clique_weights = dict()
        for clique in self.cliques:
            num_worlds = math.prod([var.domain_length for var in clique])
            self.clique_weights[clique] = fraction.Fraction(torch.zeros(size=(num_worlds,)),
                                                            torch.full((num_worlds,),denominator)).to(self.device)

    def fit(self, dataloader):
        """Fit the model to the conjunctive probability distribution of the data without having to rely 
           on gradient descent (which has to recalculate Z every once in a while.)"""

        dataset_length = len(dataloader.dataset)
        if self.verbose > 0:
            pbar = tqdm.tqdm(dataloader, desc="Fitting MRF", 
                             total=math.ceil(dataset_length / dataloader.batch_size))

        self._initialize_weights(dataset_length)

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

                    #update the clique weights
                    for clique, weights in self.clique_weights.items():
                        self.clique_weights[clique][world_begin:world_end] += fraction.Fraction(summed, torch.full(summed.shape, dataset_length)).to(self.device)
                        self.clique_weights[clique][world_begin:world_end] = self.clique_weights[clique][world_begin:world_end].simplify().to(self.device)

                    world_begin = world_end
                    
