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
    dataset = torch_mrf.alarm_dataset.AlarmDataset(10000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10000)
    
    model = torch_mrf.mrf.MarkovRandomField(dataset.random_variables, device="cuda", max_parallel_worlds=pow(2,16),
            cliques=[["Burglary", "Alarm", "Earthquake"], ["JohnCalls"], ["MaryCalls"]])

    # model = torch_mrf.mrf.MarkovRandomField(dataset.random_variables, device="cuda",max_parallel_worlds=pow(2,16),
    #         cliques=[["Burglary", "Alarm", "Earthquake","JohnCalls"],["JohnCalls", "MaryCalls"]])
    
    trainer = torch_mrf.trainer.Trainer(model, dataloader, learn_structure=True)
    model = trainer.train()

    print("Z=",model.Z)

    partial_prediction = model.predict([dict(Alarm=True, Burglary=True, Earthquake=True),
                                        dict(Alarm=False, JohnCalls=False)])
    print(list(model.parameters()))

    print(partial_prediction.float())
    model.plot()

if __name__ == '__main__':
    main()