"""This module holds implementations of Datasets which can be used for MRFs."""

import torch
import os
import pracmln
import numpy as np


class MRFDataset(torch.utils.data.Dataset):
    """A dataset for a Markov Random Field (MRF).

    This class represents the dataset that can be used to train or infer an MRF.

    Attributes:
        samples: A list of lists where the outer list describes the number of samples and
        the inner lists contain the grounded atoms which describe the samples. 
    """

    def __init__(self, path=os.path.join("..","pracmln","examples","alarm", "alarm.pracmln"),
                 mln="alarm-kreator.mln", database="query1.db", cluster_domain="person",
                 placeholder="?p"):
        """Initialize an MRFDataset.

        Loads a database as an iteratable PyTorch dataset. The databases are preprocessed s. t.
        they can be used for MRF learning and inference.
        
        Args:
            path (str): Path to the mlnproject file. This is not needed if the mln and database are passed directly.
            mln (str or pracmln.MLN): The name of the mln or the mln itself.
            database (str or pracmln.Database): The name of the database of the database itself.
            cluster_domain (str): The domain which holds information about indivduals and therefore needs to be changed.
            placeholder (str): The placeholder that will replace the individuals.
        
        Returns:
            MRFDataset: The Dataset

        """
        super(MRFDataset, self).__init__()

        if isinstance(mln, str):
            mln = pracmln.MLN.load(path + ":" + mln)

        if isinstance(database, str):
            database = pracmln.Database.load(mln, path + ":" + database)

        all_clusters = []

        #get all predicates that are not dependent on a cluster
        unaffected_predicates = [pred for pred in mln.predicates if cluster_domain not in pred.argdoms]
        unaffected_prednames = [pred.name for pred in unaffected_predicates]

        #this loop creates a list of lists, where the outer lists enumerates all clusters and
        #the inner lists contain all predicates that where true for that cluster
        for db in database:
            #get all clusters that exist in this domain
            current_clusters = db.domains[cluster_domain]
            
            #parse all evidence to literals
            current_atoms = []
            for atom in db.evidence:
                true, predname, args = mln.logic.parse_literal(atom)
                atom = mln.logic.gnd_atom(predname, args, mln)
                current_atoms.append(atom)

            for cluster in current_clusters:
                #get all atoms that describe stuff about this cluster
                related_atoms = [ca for ca in current_atoms if cluster in ca.args]
                related_atoms.extend([ca for ca in current_atoms if ca.predname in unaffected_prednames])
                
                #replace the cluster index by a generic one to compare later
                for atom in related_atoms:
                    for idx, arg in enumerate(atom.args):
                        if arg in current_clusters:
                            atom.args[idx] = placeholder

                all_clusters.append(related_atoms)

        self.samples = all_clusters
        

    def __len__(self):
        """Return the number of samples in this dataset.
        
        Returns:
            length (int): The amount of samples in this dataset.
            
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """Return the idx_th sample of this dataset.
        
        Returns:
            sample (list<pracmln.Logic.GroundAtom>): The list of grounded atoms which describe the idx_th sample.

        """
        return self.samples[idx]