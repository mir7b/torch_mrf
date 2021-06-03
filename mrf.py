import torch
import torch.nn as nn
import mrf_dataset
import os
import pracmln
class MarkovRandomField(nn.Module):
    def __init__(self, mln, cliques):
        """Constructs a Markov Random Field from the nodes and edges."""
        self.nodes = nodes
        self.edges = edges

        self.potential_functions = []

    def plot(self):
        pass
    

def main():

    path=os.path.join("..","pracmln","examples","alarm", "alarm.pracmln")
    mln_name="alarm-kreator.mln"
    db_name="query1.db"
    
    mln = pracmln.MLN.load(path + ":" + mln_name)
    database = pracmln.Database.load(mln, path + ":" + db_name)

    dataset = mrf_dataset.MRFDataset(mln=mln, database=database)

    dataloader = torch.utils.data.DataLoader(dataset)

    print(mln.domains)


    mrf = MarkovRandomField(["Burglary", "Earthquake", "Alarm", "John Calls", "Mary Calls"],
                            [("Burglary", "Earthquake"), ("Burglary", "Alarm"), ("Alarm", "Earthquake"), 
                             ("John Calls", "Alarm"), ("Mary Calls", "Alarm")])


if __name__ == "__main__":
    main()