import torch_mrf.mrf
import torch_mrf.alarm_dataset
import torch_mrf.mrf_utils
import torch_mrf.trainer
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch_random_variable.torch_random_variable import BinaryRandomVariable

def main():
    dataset = torch_mrf.alarm_dataset.AlarmDataset(pow(10,5))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=pow(10,6))
    
    model = torch_mrf.mrf.MarkovRandomField(dataset.random_variables, device="cuda", max_parallel_worlds=pow(2,16),
            cliques=[["Burglary", "Alarm", "Earthquake"], ["Alarm", "JohnCalls"], ["Alarm","MaryCalls"]], verbose=2)

    # model = torch_mrf.mrf.MarkovRandomField(dataset.random_variables, device="cuda",max_parallel_worlds=pow(2,16),
    #         cliques=[["Burglary", "Alarm", "Earthquake","JohnCalls"],["JohnCalls", "MaryCalls"]])
    
    trainer = torch_mrf.trainer.Trainer(model, dataloader, learn_structure=False)
    model = trainer.train()
    print("Z=",model.Z)
    model.Z = torch.tensor(1., device='cuda:0', dtype=torch.float64)
    prob = 0
    partial_prob = 0
    for sample in dataset.samples:
        if torch.all(sample == torch.tensor([1.,0.,0.,0.,0.], dtype=torch.bool)):
            prob += 1
        if torch.all(sample[:3] == torch.tensor([0.,0.,0.], dtype=torch.bool)):
            partial_prob += 1
            
    prob /= len(dataset)
    partial_prob /= len(dataset)
    prob_model = model.forward_no_z(torch.tensor([[1.,0.,0.,0.,0.]], dtype=torch.bool))
    print("Approximated Z via 1 world", prob_model/prob)
    partial_prediction = model.predict([dict(Alarm=False, Earthquake=False, Burglary=False)])
    print("Approximated Z via one clique",partial_prediction/partial_prob, partial_prob, partial_prediction)
    print("-------------------------------------------------------------")
    uniques, counts = torch.unique(dataset.samples, return_counts=True, dim=0)
    counts = counts.double() /  len(dataset)
    prediction = model.forward_no_z(uniques)

    error = prediction.cpu()/counts
    print(torch.mean(error), error.mean())
    #model.plot()
    exit()
    
    

if __name__ == '__main__':
    main()