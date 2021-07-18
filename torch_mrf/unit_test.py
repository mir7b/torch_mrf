import torch_mrf.mrf
import torch_mrf.alarm_dataset
import torch_mrf.mrf_utils

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch_random_variable.torch_random_variable import BinaryRandomVariable

def main():
#     dataset = torch_mrf.alarm_dataset.AlarmDataset(10000)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=10000)
    
#     model = torch_mrf.mrf.MarkovRandomField(dataset.random_variables, device="cuda", max_parallel_worlds=pow(2,16),
#             cliques=[["Burglary", "Alarm", "Earthquake"], ["Alarm", "JohnCalls"],["Alarm", "MaryCalls"]])

#     # model = torch_mrf.mrf.MarkovRandomField(dataset.random_variables, device="cuda",max_parallel_worlds=pow(2,16),
#     #         cliques=[["Burglary", "Alarm", "Earthquake","JohnCalls"],["JohnCalls", "MaryCalls"]])

#     model.fit(dataloader)
    
    rvars = [BinaryRandomVariable("Apfel"), BinaryRandomVariable("Rot"), BinaryRandomVariable("Rund"), BinaryRandomVariable("Ball")]
    model = torch_mrf.mrf.MarkovRandomField(rvars, device="cuda", max_parallel_worlds=pow(2,16),cliques=[["Apfel", "Rot"], ["Apfel" ,"Rund"], ["Ball", "Rund"]])
    pweights = torch.tensor([[0.1,0.4,0.2,0.3],[0.5,0.3,0.1,0.1], [0.35, 0.25,0.3,0.1]], dtype=torch.double, device=model.device)
    with torch.no_grad():
        for idx, (clique, weights) in enumerate(model.clique_weights.items()):
            model.clique_weights[clique].data = pweights[idx]
    print("ITS Z TIME")
    model.calc_z()

    with torch.no_grad():
        for idx, (clique, weights) in enumerate(model.clique_weights.items()):
            print(clique)
            model.clique_weights[clique].data /= 4
    
    model.calc_z()
    print("Z=",model.Z)
    print("ITS PREDICTION TIME")
    print("Prediction", model(torch.tensor([True, True, True, True], dtype=torch.bool, device=model.device).unsqueeze(0)))
    
    #print(list(model.parameters()))
    model.plot()
    exit()
    # criterion = nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1)
    

    # pbar = tqdm.tqdm(range(100), desc="Training mrf")
    # for _ in pbar:
    #     for batch in dataloader:
    #         preds = model(batch.to(model.device))

    #         loss = criterion(preds, torch.ones(size=preds.shape, dtype = torch.double, device = preds.device))
    #         loss.backward(retain_graph=True)
            
    #         optimizer.step()
    #         optimizer.zero_grad()
            
    #         model.clip_weights()
    #         model.calc_z()

    #         pbar.set_postfix(train_loss=loss.item())

    model.calc_z()
    print("Z=",model.Z)
    
    partial_prediction = model.predict([dict(Alarm=True, Burglary=True, Earthquake=True),
                                        dict(Alarm=False, JohnCalls=False)])
    # print(list(model.parameters()))

    print(partial_prediction.float())
    model.plot()

if __name__ == '__main__':
    main()