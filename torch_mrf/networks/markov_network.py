"""This file describes vecotrized Markov Random Fields for the GPU."""

from typing import List, Union
import torch
import torch.nn as nn
from torch_mrf import mrf_utils
import tqdm
import plotly.express as px
import plotly.graph_objects as go
import networkx
import itertools
import torch_random_variable.torch_random_variable as trv
from ..factors.discrete_factor import DiscreteFactor
    

class MarkovNetwork(nn.Module):
    """Represents a Markov Random Field (MRF) from a set of random variables and cliques.
    
    The MRF is highly vectorized and can be used for partial queries.

    Attributes:
        random_variables (iterable<torch_random_variable.RandomVariable>): The random variables that are represented
            in this Markov Random Field
        cliques (iterable<iterable<torch_random_variable.RandomVariable>>): The connectivity of the random variables
            as a list of lists where the inner lists represent all members of a clique. Can also be a list of lists of
            strings, where the strings are the variable names of the clique members.
        device (str): The device where the Markov Random Field will perform most of its calculations
        max_parallel_worlds (int): The maximum number of worlds that are evaluated at once on the graphics card.
            This parameter can produce an Cuda Out of Memory Error when picked too large and slow down speed when picked too small.
        verbose (bool): Whether to show progression bars or not
        Z (torch.Tensor<torch.double>): The overall probability mass.
        universe_matrix (torch.Tensor<torch.bool>): The whole universe that can be created from all random variables of the MRF.
        clique_universes (dict<frozenlist<torch_random_variable.RandomVariable>, torch.Tensor>): The universes
            that get covered by each clique.
        clique_weights (dict<str, torch.nn.parameter.Parameter>): A dict that maps the cliques to the weights for 
            each clique which will be optimized.
        
    """

    def __init__(self, random_variables:List[trv.RandomVariable], cliques:List[List[Union[str, trv.RandomVariable]]],
                factor = DiscreteFactor, device:str or int="cuda", max_parallel_worlds:int = pow(2,20),verbose:int=1):
        """Construct a Markov Random Field from the nodes and edges.

        Args:
            random_variables (iterable<torch_random_variable.RandomVariable>): The random variables that are represented
                in this Markov Random Field
            cliques (iterable<iterable<torch_random_variable.RandomVariable>>): The connectivity of the random variables
                as a list of lists where the inner lists represent all members of a clique
            device (str): The device where the Markov Random Field will perform most of its calculations
            max_parallel_worlds (int): The maximum number of worlds that are evaluated at once on the graphics card.
                This parameter can produce an Cuda Out of Memory Error when picked too large and slow down speed when picked too small.
            verbose (int): Level of verbosity

        """
        super(MarkovNetwork, self).__init__()
        
        self.random_variables:List[trv.RandomVariable] = random_variables
        self.verbose:int = verbose
        self.device:str or int = device
        self.max_parallel_worlds:int = max_parallel_worlds

        #parse clique members to variable if they arent already variables
        for idx, clique in enumerate(cliques):
            for jdx, partner in enumerate(clique):
                if isinstance(partner, str):
                    corresponding_random_variables = [var for var in self.random_variables if var.name==partner]
                    #check if the partner is part of this mrfs random variables
                    if len(corresponding_random_variables) == 0:
                        raise Exception("Random variable name %s was used in a clique, but it does not exist in\
                                        the MRF. \n Random Variable names in this MRF: %s"\
                                        % (partner, [var.name for var in self.random_variables]))

                    cliques[idx][jdx] = corresponding_random_variables[0]
        
        self.cliques:nn.ModuleList[DiscreteFactor] = nn.ModuleList()
        
        for clique in cliques:
            phi = factor(clique, self.device, self.max_parallel_worlds, verbosity=self.verbose)
            self.cliques.append(phi)
        
        self.cliques_rescalings:torch.Tesnor = torch.ones((len(self.cliques)))
        
        self.Z = torch.tensor(1, dtype=torch.double)

    def _get_rows_in_universe(self, random_variables:List[trv.RandomVariable]):
        """Get the row indices of the random_variables in the universe.
        
        Args:
            random_variables (iterable<torch_random_variable.RandomVariable>): The random variables which row
            slices are desired

        Returns:
            rows (torch.Tenosr<torch.long>): A tensor that contains the slices of the variables.
        """
        rows = [-1] * len(random_variables)
        
        for idx, random_variable in enumerate(self.random_variables):
            for jdx, random_variable_ in enumerate(random_variables):
                if random_variable == random_variable_:
                    rows[jdx] = idx
        return rows


    def forward(self, x):
        return self.forward_no_z(x) / self.Z


    def forward_no_z(self, x):
        probs = torch.ones((len(x),), device = self.device, dtype = torch.float)
        for idx, clique in enumerate(self.cliques):
            rows = self._get_rows_in_universe(clique.random_variables)
            potential = clique(x[:,rows]) * self.cliques_rescalings[idx]
            probs = probs * potential
        return probs


    def fit(self, x, calc_z=False, rescale_weights=True):
        with torch.no_grad():
            for idx, clique in tqdm.tqdm(enumerate(self.cliques), desc="Fitting model parameters", total=len(self.cliques)) \
                if self.verbose > 0 else enumerate(self.cliques):
                rows = self._get_rows_in_universe(clique.random_variables)
                clique.fit(x[:,rows])
                
                if rescale_weights:
                    #number of trainable parameters
                    notp = sum([p.numel() for p in clique.parameters() if p.requires_grad])
                    self.cliques_rescalings[idx] = notp/2
                
            if calc_z:
                self.calc_z()
            
    def calc_z(self, set_Z:bool = True):
        """Calculate the probability mass of this mrf with respect to the current weights."""
        with torch.no_grad():
            #initialize new Z
            Z = torch.tensor(0, dtype=torch.double, device=self.device)

            #iterate over world batches
            for world_batch in mrf_utils.iter_universe_batches_(self.random_variables, max_worlds=self.max_parallel_worlds, 
                                                            verbose=self.verbose > 0):

                #calc their probability mass
                probabilities = self.forward_no_z(world_batch.to(self.device))

                #add up the new overall probability mass
                Z += torch.sum(probabilities)
                
            #set it as class variable
            if set_Z:
                self.Z = Z
                
            return Z

    def plot_structure(self, layout = networkx.drawing.layout.spring_layout) -> go.Figure:
        
        #generate nodes and edges of the networkx graph
        nodes = set([var.name for var in self.random_variables])
        edges = []
        for clique in self.cliques:
            names = [var.name for var in clique.random_variables ]
            edges.extend(list(itertools.product(names, names)))
        
        #create the graph
        graph = networkx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        
        #apply a layout
        pos = layout(graph)

        edge_x = []
        edge_y = []
        for edge in graph.edges():
            x0,y0 = pos[edge[0]]
            x1,y1 = pos[edge[1]]

            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        node_adjacencies = []
        node_text = []
        
        for node, adjacencies in enumerate(graph.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append("Variable Name: " + str(list(graph.nodes())[node]))

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Structure of Markov Random Field',
                #titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                #margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
        return fig
     