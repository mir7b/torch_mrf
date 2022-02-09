import torch
import torch_random_variable.torch_random_variable as trv
from torch_mrf.networks.markov_network import MarkovNetwork
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots

def mnist():
    from sklearn.datasets import load_digits
    import sklearn.metrics
    from torchvision import transforms
    X,y = load_digits(return_X_y = True)
    
    rvars = []
    for i in range(8):
        for j in range(8):
            rvars.append(trv.RandomVariable("%s%s" % (i,j), list(range(7))))
    rvars.append(trv.RandomVariable("Digit", list(range(10))))
    
    X:torch.Tensor = torch.tensor(X)
    original_data = X
    X = X/3
    X = X.long()
    y = torch.tensor(y).unsqueeze(-1).long()

    cliques = []
    for i in range(8):
        for j in range(8):
            clique = ["%s%s" % (i,j), "Digit"]
            if i < 7:
                cliques.append( clique + ["%s%s" % (i+1,j)])
            if j < 7:
                cliques.append( clique + ["%s%s" % (i,j+1)])


    data = torch.cat((X,y),dim=-1)
    model = MarkovNetwork(rvars, cliques, device="cuda")
    model.fit(data, calc_z=False)

    prediction = torch.zeros(y.shape)
    
    queries = X.repeat_interleave(10,0)
    classes = torch.arange(0,10).repeat(len(X),1).flatten().unsqueeze(-1).long()
    queries = torch.cat((queries, classes), dim=-1)
    probability = model(queries)
    probability = probability.reshape(len(X),10).cpu().detach()
    prediction = torch.argmax(probability, dim=1).long()

    
    print(sklearn.metrics.confusion_matrix(y,prediction.numpy()))
    print(sklearn.metrics.accuracy_score(y,prediction.numpy()))

    missclassifications = prediction != y.squeeze()
    missclassifications = missclassifications.nonzero()
    
    if len(missclassifications) == 0:
        exit()
        
    fig = plotly.subplots.make_subplots(rows=len(missclassifications), cols=2, 
                                        column_titles=["Missclassified Image", "Probility Distribution"],
                                        row_titles= ["True Class: %s" % y[idx].item() for idx in missclassifications],
                                        column_widths=[400,400])

    for idx in range(len(missclassifications)):
        fig.add_trace(go.Bar(x = list(range(10)), y=probability[missclassifications[idx]].squeeze()),idx+1,2)
        fig.add_trace(go.Heatmap(z=original_data[missclassifications[idx]].reshape(8,8), colorscale="gray", showscale=False),idx+1,1)
        
    fig.update_layout(showlegend=False, height=400*len(missclassifications), width=800, title="Evaluation of an MRF for the MNIST dataset")
    fig.show()

    
def main():
    mnist()

if __name__ == "__main__":
    main()