import torch
import torchvision.datasets
import torch.utils.data.dataloader
import torch.utils.data.dataset
from torchvision import transforms
import torch_random_variable.torch_random_variable as trv
from torch_mrf.networks.markov_network import MarkovNetwork
from torch_mrf.factors.discrete_factor import DiscreteFactor
import os
import plotly.graph_objects as go
import sklearn.metrics

class JointCIFAR10(torch.utils.data.dataset.Dataset):
    def __init__(self,cifar10):
        super(JointCIFAR10, self).__init__()
        self.cifar10 = cifar10
        
    def __getitem__(self,idx):
        img, clazz = self.cifar10[idx]
        img = transforms.ToTensor()(img).flatten()
        img = img * 255
        clazz = torch.tensor(clazz).unsqueeze(0)
        sample = torch.cat((img, clazz))
        return sample.long()
    
    def __len__(self):
        return len(self.cifar10)
        
        

cifar = torchvision.datasets.CIFAR10(root=os.path.join("..", "Downloads"), download=True)
dataset = JointCIFAR10(cifar)
dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=1000)

random_variables = []
cliques = []

for i in range(32):
    for j in range(32):
        for k in range(3):
            random_variables.append(trv.RandomVariable(str((i,j,k)), list(range(256))))
            cliques.append([str((i,j,k)), "Label"])

random_variables.append(trv.RandomVariable("Label", list(range(10))))
model = MarkovNetwork(random_variables, cliques, verbose=1)

for batch in dataloader:
    model.fit(batch)
    model.eval()
    queries = batch.repeat_interleave(10,0)[:,:-1]
    classes = torch.arange(0,10).repeat(len(batch),1).flatten().unsqueeze(-1).long()
    queries = torch.cat((queries, classes), dim=-1)
    probability = model(queries, discriminative=True)
    probability = probability.reshape(len(batch),10).cpu().detach()
    prediction = torch.argmax(probability, dim=1).long()

    cm = sklearn.metrics.confusion_matrix(batch[:,-1],prediction.numpy())
    print(cm)
    exit()