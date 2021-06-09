import pytorch_lightning as pl
from torch import optim
import mrf
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import pracmln
import mrf_dataset
from torch_random_variable import RandomVariable

class LightningMarkovRandomField(pl.LightningModule):
    def __init__(self, random_variables, cliques, device="cuda", max_parallel_worlds = 1024,verbose=True):
        super(LightningMarkovRandomField, self).__init__()
        self.mrf = mrf.MarkovRandomField( random_variables, cliques, device="cuda", max_parallel_worlds = 1024,verbose=True)

    def forward(self, tensor):
        return self.mrf(tensor)

    def training_step(self, batch, batch_idx):
        prediction = self.mrf(batch)
        loss = F.binary_cross_entropy(prediction, torch.ones(size=prediction.shape, dtype = torch.double, device = prediction.get_device()))
        self.manual_backward(loss, retain_graph=True)
        self.log("Train Loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        return optimizer

    

def main():
    path=os.path.join("..","pracmln","examples","alarm", "alarm.pracmln")
    mln_name="alarm-kreator.mln"
    db_name="query1.db"
    mln = pracmln.MLN.load(path + ":" + mln_name)
    database = pracmln.Database.load(mln, path + ":" + db_name)

    random_variables = [RandomVariable(name,domain) for name, domain in mln.domains.items() if name!="person"]

    mrf = LightningMarkovRandomField(random_variables, [["domNeighborhood","place"]])

    dataset = mrf_dataset.MRFDataset(mln=mln, database=database, random_variables=random_variables)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

    trainer = pl.Trainer(gpus=1)

    trainer.fit(mrf, dataloader)

if __name__ == "__main__":
    main()