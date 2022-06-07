from os import name
from numpy import var
import torch
import torch.nn as nn
from typing import List
import torch_random_variable.torch_random_variable as trv
import plotly.graph_objects as go
import tqdm
import jpt.trees
from jpt.variables import NumericVariable, SymbolicVariable, Variable
from jpt.learning.distributions import Numeric, SymbolicType
import jpt_extensions.parallel_inference
import plotly.subplots
import numpy as np
import networkx

class JPTFactor(nn.Module):
    """
    A discrete factor function that maps every world that is possible by its random variables to a real positive number.
    
    :param random_variables: The random variables that are used by this factor.
    :type random_variables: List of torch variables
    :param verbose: The verbosity level
    :type verbose: int
    :param device: The hardware device the factor will do its calculations on
    :type device: str or int
    :param max_parallel_worlds: The maximum number of parallel worlds that will be used by this factor. Setting this parameter too
        low can take alot of time. Setting it too high can cause a memory problem.
    :type max_parallel_worlds: int
    :param weights: The weights that are used to calculate the potential of a state.
    :type weights: torch.tensor
    """
    
    def __init__(self,random_variables:List[trv.RandomVariable], device:str or int="cpu",
                 max_parallel_worlds:int = pow(2,20), fill_value = 0., verbosity:int=1):
        """Create a factor that describes the potential of each state. This factor has exponential many parameters in the
        number of random variables.

        Args:
            random_variables (List[trv.RandomVariable]): The involved random variables.
            device (str or int, optional): Device the factor will lay on. Defaults to "cuda".
            max_parallel_worlds (int, optional): The maximum number of parallel worlds that will be used by this factor. 
                Setting this parameter too low can take alot of time. Setting it too high can cause a memory problem. 
                Defaults to pow(2,20).
            fill_value ([type], optional): The default value for each weight in the factor. Defaults to 1..
            verbose (int, optional): The verbosity level of this factor. Defaults to 1.
        """
    
        super(JPTFactor, self).__init__()
        
        self.random_variables:List[trv.RandomVariable] = random_variables
        self.verbosity:int = verbosity
        self.device:str or int = device
        self.max_parallel_worlds:int = max_parallel_worlds

        jpt_variables = []

        #convert torch random variables to jpt random variables
        for variable in self.random_variables:

            if isinstance(variable, trv.NumericRandomVariable):
                variable_ = NumericVariable(variable.name)

            else:
                discrete_type = SymbolicType(variable.name, variable.domain)
                variable_ = SymbolicVariable(variable.name, domain=discrete_type)

            jpt_variables.append(variable_)


        self.tree = jpt.trees.JPT(jpt_variables, min_samples_leaf=1)
    
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """ Takes a tensor with shape (samples, len(self.random_variables)) and returns the potential of each sample.

        Args:
            x (torch.Tensor): the samples.
        Returns:
            torch.Tensor: the potential
        """
        probs =  jpt_extensions.parallel_inference.infer_parallel(self.tree, x.double().numpy())
        return torch.tensor(probs)
    
    def fit(self, data:torch.Tensor):
        """Calculates weights for this factor by infering the probability of each sample in the provided data.
        This is optimal w. r. t. the parameters as proven in TODO.

        Args:
            data (torch.Tensor): The data that is observed for this factor. data is a tensor of shape 
                                 (number of observations ,number of random_variables in this factor)) 
                                 and and indexable type like long or bool etc.
        """
        self.tree.learn(data.numpy())


    def plot(self) -> go.Figure:
        """Visaulizes the joint probability tree that represents this factor.

        Returns:
            None. The plot will be shown automatically
        """
        fig:go.Figure = plotly.subplots.make_subplots(rows=len(self.tree.leaves), 
            cols = len(self.random_variables), column_titles=[v.name for v in self.random_variables],
            row_titles=[leaf.format_path() for leaf in self.tree.leaves.values()])


        for row, (id, leaf) in enumerate(self.tree.leaves.items()):
            leaf:jpt.trees.Leaf = leaf
            for column, (variable, distribution) in enumerate(leaf.distributions.items()):
                variable:Variable = variable
                if variable.symbolic:
                    fig.add_trace(go.Bar(x=list(distribution.labels),
                                         y=distribution._params),
                                    row=row+1, col=column+1)
                else:
                    std = (np.std([i.upper - i.lower for i in distribution.cdf.intervals[1:-1]]) or
                                distribution.cdf.intervals[1].upper - distribution.cdf.intervals[1].lower) * 2
                    bounds = np.array([distribution.cdf.intervals[0].upper - std / 2] +
                                    [v.upper for v in distribution.cdf.intervals[:-2]] +
                                    [distribution.cdf.intervals[-1].lower] +
                                    [distribution.cdf.intervals[-1].lower + std / 2])

                    bounds_ = np.array([distribution.labels[b] for b in bounds])
                    fig.add_trace(go.Scatter(x=bounds_, y=np.asarray(distribution.cdf.multi_eval(bounds))), row=row+1, col=column+1)


        fig.update_layout(showlegend=False, height=400*len(self.tree.leaves), 
            width=800, title="Distributions in JPT for clique %s" % self.random_variables)

        return fig


    def plot_tree(self) -> go.Figure:
        nodes = [node.idx for node in self.tree.allnodes.values()]
        edges = [(node.parent.idx, node.idx) for node in self.tree.allnodes.values() 
                    if node.parent]
        
        graph = networkx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        
        positions =  networkx.drawing.nx_agraph.graphviz_layout(graph, prog='dot')

        node_trace = go.Scatter(x = [x for (x,_) in positions.values()],
                                y = [y for (_,y) in positions.values()],
                                mode="markers+text", name="Nodes",
                                textposition="top right",
                                text=["Node %s" % idx for idx in nodes],
                                marker=dict(
                                    color='LightSkyBlue',
                                    size=20,))

        edge_trace_x = []
        edge_trace_y = []
        edge_trace_text = []
        for a,b in edges:
            x1,y1 = positions[a]
            x2,y2 = positions[b]
            edge_trace_x.extend([x1, (x1+x2)/2, x2,None])
            edge_trace_y.extend([y1, (y1+y2)/2, y2,None])
            edge_trace_text.extend(["", self.tree.allnodes[b]._path[-1][1], "", ""])

        edge_trace = go.Scatter(x=edge_trace_x, y=edge_trace_y, mode="lines+text",
            line=dict(color='darkgray', width=4), name="Edges", text=edge_trace_text,
                textposition="top right")

        fig = go.Figure()
        fig.add_trace(edge_trace)
        fig.add_trace(node_trace)

        fig.update_layout(title = "JPT for clique %s" % self.random_variables,
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

        return fig