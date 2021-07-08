import torch_mrf.mrf
import torch_mrf.alarm_dataset

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

def main():
    dataset = torch_mrf.alarm_dataset.AlarmDataset(10000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10000)
    
    # model = torch_mrf.mrf.MarkovRandomField(dataset.random_variables, device="cuda", max_parallel_worlds=pow(2,16),
    #         cliques=[["Burglary", "Alarm", "Earthquake"], ["Alarm", "JohnCalls"], ["Alarm", "MaryCalls"]])
    model = torch_mrf.mrf.MarkovRandomField(dataset.random_variables, device="cuda",max_parallel_worlds=pow(2,16),
            cliques=[["Burglary", "Alarm", "Earthquake","JohnCalls"],["JohnCalls", "MaryCalls"]])

    model.fit(dataloader)

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

if __name__ == '__main__':
    main()