from typing import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GINEConv, GINConv


class GNN(nn.Module):
    def __init__(self, num_of_features, device):
        super(GNN, self).__init__()
        self.num_of_features = num_of_features
        # Define the dimensions of 4-layer GCN
        self.gcn_dimensions = [128] * 4  # Keep each layer's dimension as 128
        self.SOPOOL_dim_1 = 32
        self.SOPOOL_dim_2 = 32
        self.linear_hidden_dimensions = 32
        self.output_dimensions = 2
        
        self.device = device
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(num_of_features, self.gcn_dimensions[0]))

        for i in range(1, 4):  
            self.gcn_layers.append(GCNConv(self.gcn_dimensions[i-1], self.gcn_dimensions[i]))

        self.SOPOOL = nn.Sequential(OrderedDict([
              ("Linear_1", nn.Linear(self.gcn_dimensions[-1], self.SOPOOL_dim_1)),
              ("ReLU", nn.ReLU()),
              ("Linear_2", nn.Linear(self.SOPOOL_dim_1, self.SOPOOL_dim_1)),
              ("ReLU", nn.ReLU()),
              ("Linear_3", nn.Linear(self.SOPOOL_dim_1, self.SOPOOL_dim_2)),
              ("ReLU", nn.ReLU())
        ]))
        
        self.MLP_1 = nn.Sequential(OrderedDict([
              ("Linear_1", nn.Linear(self.SOPOOL_dim_2 ** 2, self.linear_hidden_dimensions)),
              ("ReLU", nn.ReLU()),
              ("Linear_2", nn.Linear(self.linear_hidden_dimensions, self.linear_hidden_dimensions)),
              ("ReLU", nn.ReLU()),
              ("Linear_3", nn.Linear(self.linear_hidden_dimensions, self.output_dimensions)),
              ("ReLU", nn.ReLU()),
        ]))

    def forward(self, graph_batch, edge_weight=None):
        """
        :param graph_batch: Graph batch object from PyTorch Geometric containing node features, edge indices, etc.
        :param edge_weight: Optional edge weight (could be the edge probability matrix)
        """
        graph_batch = graph_batch.to(self.device)
        
        # Use edge_attr as the weight
        edge_weights = graph_batch.edge_attr
        if edge_weights is not None:
            # Only take the first column as the weight (retain the probability of the edge existing)
            edge_weights = edge_weights[:, 0].squeeze()

        
        x = graph_batch.x
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x=x,
                         edge_index=graph_batch.edge_index,
                         edge_weight=edge_weights)
            x = F.relu(x)
        
        node_features_ = F.dropout(x, p=0.5, training=self.training)
        normalized_node_features = F.normalize(node_features_, dim=1)
        
        def sep_graph(node_features, ptr):
            graphs = []
            for i in range(len(ptr)-1):
                temp1 = ptr[i]
                temp2 = ptr[i+1]
                graphs.append(node_features[temp1:temp2])
            return(graphs)

        normalized_node_features = sep_graph(normalized_node_features, graph_batch.ptr)

        HH_tensor = torch.Tensor()

        for graph in normalized_node_features:
            graph = self.SOPOOL(graph)
            temp = torch.mm(graph.t(), graph).view(1, -1)
            if HH_tensor.shape == (0,):
                HH_tensor = temp
            else:
                HH_tensor = torch.cat((HH_tensor, temp), dim=0)

        output = F.dropout(self.MLP_1(HH_tensor), p=0.5, training=self.training)
        
        torch.cuda.empty_cache()
        return HH_tensor, output