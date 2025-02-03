import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch_geometric.data import Data


class MLP_subgraph(nn.Module):
    def __init__(self, node_features_num, edge_features_num, device):
        super(MLP_subgraph, self).__init__()
        self.device = device
        self.node_features_num = node_features_num
        self.edge_features_num = edge_features_num
        self.mseloss = torch.nn.MSELoss()
        self.feature_size = 64
        self.linear = nn.Linear(self.node_features_num, self.feature_size).to(
            self.device
        )
        self.linear1 = nn.Linear(2 * self.feature_size, 8).to(self.device)
        self.linear2 = nn.Linear(8, 1).to(self.device)

        # Random tensor with uniform distribution
        self.node_feature_mask = nn.Parameter(torch.rand(node_features_num))

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size()) + (1 - bias)
            gate_inputs = (torch.log(eps) - torch.log(1 - eps)).to(
                self.device
            )  # Logit Function
            gate_inputs = gate_inputs.to(self.device)  # move to device
            # print(f'\ntemperature{temperature.device}')
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph
    
    def _concrete_sample(self, log_alpha, beta=1.0, training=True):
        '''gate_inputs = (GumbelNoise + log_alpha) / beta'''
        if training:
            random_noise = torch.rand(log_alpha.shape)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()
        return gate_inputs

    def _node_feature_mask(self, graph):
        """
        Calculate the node feature mask
        """
        mask = self.node_feature_mask

        # Weight the features of each node, and the weight is provided by the mask.
        node_prob_matrix = graph.x * mask

        # Input the node probability matrix into the MLP for computation
        x = self.linear(node_prob_matrix).to(self.device)
        f1 = x.unsqueeze(1).repeat(1, 105, 1).view(-1, self.feature_size)
        f2 = x.unsqueeze(0).repeat(105, 1, 1).view(-1, self.feature_size)
        f12self = torch.cat([f1, f2], dim=-1)


        f12self = torch.sigmoid(self.linear2(torch.sigmoid(self.linear1(f12self))))

        mask_sigmoid = f12self.reshape(105, 105)


        # A symmetric edge probability matrix "sys_mask"
        # Symmetric edge probability matrix
        sys_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2

        # Edge Probability
        edge_prob = sys_mask[graph.edge_index[0], graph.edge_index[1]]
        
        # two methods to sample the edge probability
        # edge_prob = self._sample_graph(
        #     edge_prob, temperature=0.5, bias=0.0, training=self.training
        # )
        edge_prob = self._concrete_sample(edge_prob, beta=0.5, training=self.training)
        # edge_prob shape:  [E] -- > [E,2]
        p_complement = 1 - edge_prob
        edge_prob = torch.stack([edge_prob, p_complement], dim=1)
        return edge_prob

    def forward(self, graph):
        subgraph = graph.to(self.device)
        edge_probs_matrix = self._node_feature_mask(subgraph)

        # Update the edge properties of the subgraph
        subgraph.edge_attr = edge_probs_matrix

        """
        The position penalty (pos_penalty) is obtained by calculating the variance of the edge probability matrix. 
        A larger variance means a more dispersed distribution of edge probabilities, which may indicate structural changes or anomalies in the graph.
        """
        pos_penalty = edge_probs_matrix.var()

        return subgraph, pos_penalty
