import os
import scipy.io as scio
from scipy.sparse import coo_matrix

import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def read_dataset():
    """
    Read data from BSNIP dataset and reconstruct data as 'torch_geometric.data.Data'
    """
    data = scio.loadmat('/Users/jingqi/Desktop/MICCAI/BSNIP/BSNIP.mat')
    dataset = []
    
    # Get BSNIP data and group information
    data = data['BSNIP']
    group = data['group']
    num = len(group)
    
    for graph_index in range(num):
        group_label = group[graph_index][0][0]
        if(group_label=='SZ' or group_label=='NC'):
            # Extract features and edges
            ROI = torch.Tensor(data[graph_index]['FC'][0])
            edge = torch.Tensor(data[graph_index]['edges'][0])
            
            # Convert label to binary (SZ: 1, NC: 0)
            label = 1 if group_label == 'SZ' else 0
            y = torch.Tensor([label])
            
            # Create sparse adjacency matrix
            A = torch.sparse_coo_tensor(
                indices=edge[:, :2].t().long(),
                values=edge[:, -1].reshape(-1,).float(),
                size=(105, 105)
            )
            G = (A.t() + A).coalesce()
            
            # Create graph data object
            graph = Data(x=ROI.reshape(-1, 105).float(),
                        edge_index=G.indices().reshape(2, -1).long(),
                        edge_attr=G.values().reshape(-1, 1).float(),
                        y=y.long())
            dataset.append(graph)
            
    print("finish read Schizophrenia dataset")
    return dataset

if __name__ == '__main__':
    dataset = read_dataset()
    print(len(dataset))
    loader = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    print(loader.y)
