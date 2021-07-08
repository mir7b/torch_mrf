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

    def _weight_collapsed_batch(self, weights, collapsed):
        #get number of batches
        _,b = collapsed.shape
        #get the weight of each holding clique
        clique_weights = weights.unsqueeze(1).repeat(1,b) * collapsed 
        return clique_weights


    def predict(self, samples):
        b = len(samples)
        #construct matrix for later satisfaction checking
        filled_samples = torch.ones((b,sum([var.encoding_length for var in self.random_variables])))

        #convert samples to variables if not yet done
        for idx, sample in enumerate(samples):
            sample_ = dict()
            for var, value in sample.items():
                if isinstance(var,str):
                    var, = [rv for rv in self.random_variables if rv.name == var]
                sample_[var] = value
                universe_rows = self._get_rows_in_universe([var])
                if len(universe_rows) == 1:
                    universe_rows = universe_rows.item()
                filled_samples[idx, universe_rows] = var.encode(value)
            samples[idx] = sample_

        #vector to store all probabilites
        probabilities = fraction.Fraction(torch.ones(size=(b,)), torch.ones(size=(b,))).to(self.device)
        sample_rows = torch.cat([self._get_rows_in_universe(list(sample.keys())).unsqueeze(0) for sample in samples],dim=0)

        
        for clique in self.cliques:
            rows = self.clique_rows_in_univserse[clique]
            clique_probabilities = fraction.Fraction(torch.zeros(size=(b,)), torch.ones(size=(b,))).to(self.device)
            for idx, sample in enumerate(samples):
                filled_sample = filled_samples[idx]
                relevant_rows = torch.cat([element.unsqueeze(0) for element in rows if element in sample_rows[idx]])
                
                if len(relevant_rows) > 0:
                    rows_in_clique = self._get_rows_in_clique(clique,[var for var in clique if var in sample])

                    world_begin = 0
                    for world_batch in mrf_utils.iter_universe_batches(list(clique),self.max_parallel_worlds,
                                                                        verbose = self.verbose > 1):
                        num_worlds = len(world_batch)

                        #track beginning of world and ending of world to use the right weights.
                        world_end = world_begin + num_worlds

                        world_batch = world_batch[:,rows_in_clique].to(self.device)
                        sample_batch = filled_sample[relevant_rows].to(self.device).unsqueeze(0).repeat(num_worlds,1)

                        true_groundings = torch.where(world_batch == sample_batch,True,False)

                        collapsed = mrf_utils.batch_collapse_sideways(true_groundings.unsqueeze(1)).squeeze(-1)

                        
                        weights = self.clique_weights[clique][world_begin:world_end]
                        weights = weights * collapsed

                        clique_probabilities[idx] += weights.sum()
                        world_begin = world_end

            probabilities *= clique_probabilities
            probabilities = probabilities.simplify()
        return probabilities


    def forward(self, samples):

        #get batch dimension
        b, _ = samples.shape

        #construct result probability masses as neutral element of multiplication
        world_probability_masses = torch.ones(size=(b,), dtype=torch.double, device=self.device)

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
                    clique_weights = self.clique_weights[clique][world_begin:world_end]

                    #get the weight of each holding clique
                    clique_weights = self._weight_collapsed_batch(clique_weights, collapsed)
                    
                    #clique_weights = collapsed * torch.repeat_interleave(clique_weights.unsqueeze(1), b, 1)

                    #sum up the weights
                    clique_weights = clique_weights.sum(dim=-2)

                    #multiply with the weight of each previous clique
                    world_probability_mass = clique_weights * world_probability_masses
                    world_begin = world_end
                    
        #return probabilities
        return world_probability_mass


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
                    self.clique_weights[clique][world_begin:world_end] += fraction.Fraction(summed, torch.full(summed.shape, dataset_length)).to(self.device)
                    self.clique_weights[clique][world_begin:world_end] = self.clique_weights[clique][world_begin:world_end].simplify().to(self.device)

                    world_begin = world_end
                    
