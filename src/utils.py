from tqdm import tqdm
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataLoader

import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split

from src.mutual_information import RenyiEntropy
from src.mutual_information import CS_QMI

def separate_data(graph_list, seed, fold_idx):
    """
    Separate the dataset into trainsets and testsets (list of graph)
    """
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    labels = [graph.y.numpy() for graph in graph_list]
    idx_list = []  # Contains 10 tuples, each containing two arrays for training and testing indices
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]  # idx_list is a list containing 10 tuples (because it is 10-fold cross-validation)
    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]
    return train_graph_list, test_graph_list


def GaussianMatrix(X,Y,sigma):
    size1 = X.size()
    size2 = Y.size()
    G = (X*X).sum(-1)
    H = (Y*Y).sum(-1)
    Q = G.unsqueeze(-1).repeat(1,size2[0])
    R = H.unsqueeze(-1).T.repeat(size1[0],1)
    H = Q + R - 2*X@(Y.T)
    gram_matrix = torch.clamp(torch.exp(-H/2/sigma**2),min=0)
    
    return gram_matrix


def calculate_MI(x, y, s_x, s_y):
    """
    Function for computing mutual information using CS_QMI method
    Args:
        x: input tensor
        y: input tensor 
        s_x: sigma for x kernel
        s_y: sigma for y kernel
    Returns:
        Mutual information calculated using CS_QMI
    """
    N = x.shape[0]
    
    # Compute two kernel matrices using the provided s_x and s_y
    Kx = GaussianMatrix(x, x, sigma=s_x)
    Ky = GaussianMatrix(y, y, sigma=s_y)
    
    # Calculate the three terms of CS_QMI
    self_term1 = torch.trace(Kx@Ky.T)/(N**2)
    self_term2 = (torch.sum(Kx)*torch.sum(Ky))/(N**4)
    
    term_a = torch.ones(1,N).to(x.device)
    term_b = torch.ones(N,1).to(x.device)
    cross_term = (term_a@Kx.T@Ky@term_b)/(N**3)
    
    mi = -2*torch.log2(cross_term) + torch.log2(self_term1) + torch.log2(self_term2)
    
    return mi


def train(args, model, train_dataset, optimizer, epoch, SG_model, device, criterion=nn.CrossEntropyLoss()):
    """
    A function used to train the model that feeds all the training data into the model once per execution
    """
    # model.train()
    # SG_model.train()
    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')
    loss_accum = 0
    miloss_accum = 0
    total_time = 0
    
    # Print basic information about the training dataset
    print(f"\nTraining dataset information:")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of iterations per epoch: {total_iters}")
    
    for pos in pbar:
        indices = range(0, len(train_dataset), args.batch_size)

        for i in indices:
            model.train()
            SG_model.train()
            graphs = train_dataset[i: i + args.batch_size]
            batch_graph = next(iter(DataLoader(graphs, batch_size=len(graphs))))
            
            # Check batch data
            if pos == 0 and i == 0:  # Print only for the first batch
                print(f"\nFirst batch data information:")
                print(f"Number of graphs in the batch: {len(graphs)}")
                print(f"Node feature shape: {batch_graph.x.shape}")
                print(f"Edge index shape: {batch_graph.edge_index.shape}")
                print(f"Edge attribute shape: {batch_graph.edge_attr.shape}")
                print(f"Label shape: {batch_graph.y.shape}")

            embeddings, original_output = model(batch_graph)
            
            # Check embeddings
            if pos == 0 and i == 0:  # Print only for the first batch
                print(f"\nEmbedding vector information:")
                print(f"Embedding vector shape: {embeddings.shape}")
                print(f"Contains NaN: {torch.isnan(embeddings).any()}")
                print(f"Value range: [{embeddings.min().item():.4f}, {embeddings.max().item():.4f}]")

            subgraphs = copy.deepcopy(graphs)
            positive_penalty = torch.Tensor([0.0]).float().to(device)

            for i in range(len(subgraphs)):
                subgraph, pos = SG_model(subgraphs[i])
                subgraphs[i] = subgraph.to(device)
                positive_penalty += pos
                
            positive_penalty = (positive_penalty / len(subgraphs))

            batch_subgraph = next(iter(DataLoader(subgraphs, batch_size=len(subgraphs))))
            positive, subgraph_output = model(batch_subgraph)

            # Check the positive sample embedding vector
            if pos == 0 and i == 0:  # Only print for the first batch
                print(f"\nPositive sample embedding vector information:")
                print(f"Shape: {positive.shape}")
                print(f"Contains NaN: {torch.isnan(positive).any()}")
                print(f"Value range: [{positive.min().item():.4f}, {positive.max().item():.4f}]")

            # Calculate to sigma1 and sigma2
            with torch.no_grad():
                Z_numpy1 = embeddings.cpu().detach().numpy()
                k = squareform(pdist(Z_numpy1, 'euclidean'))
                k = k[~np.eye(k.shape[0], dtype=bool)].reshape(k.shape[0], -1)
                
                # Check the distance matrix
                if pos == 0 and i == 0:  # Only print for the first batch
                    print(f"\nDistance matrix information:")
                    print(f"Distance matrix shape: {k.shape}")
                    print(f"Contains NaN: {np.isnan(k).any()}")
                    if k.size > 0:
                        print(f"Value range: [{k.min():.4f}, {k.max():.4f}]")
                
                # Original code:
                # sigma1 = np.mean(np.sort(k[:, :10], 1))
                # New code: Add safety check to avoid empty array issue
                if k.size > 0:
                    k_sorted = np.sort(k[:, :min(10, k.shape[1])], 1)
                    if k_sorted.size > 0:
                        sigma1 = np.mean(k_sorted)
                    else:
                        sigma1 = 0.1  # Change default value to 0.1
                else:
                    sigma1 = 0.1  # Change default value to 0.1

            with torch.no_grad():
                Z_numpy2 = positive.cpu().detach().numpy()
                k = squareform(pdist(Z_numpy2, 'euclidean'))
                k = k[~np.eye(k.shape[0], dtype=bool)].reshape(k.shape[0], -1)
                # Original code:
                # sigma2 = np.mean(np.sort(k[:, :10], 1))
                # New code: Add safety check to avoid empty array issue
                if k.size > 0:
                    k_sorted = np.sort(k[:, :min(10, k.shape[1])], 1)
                    if k_sorted.size > 0:
                        sigma2 = np.mean(k_sorted)
                    else:
                        sigma2 = 0.1  # Change default value to 0.1
                else:
                    sigma2 = 0.1  # Change default value to 0.1

            # Calculate mutual information by using RenyiEntropy class
            renyi_entropy = RenyiEntropy()
            mi_loss = renyi_entropy.calculate_mi(embeddings, positive, sigma1 ** 2, sigma2 ** 2) / len(graphs)
            
            # Using CS_QMI class
            # cs_qmi = CS_QMI()
            # mi_loss = cs_qmi.calculate_mi(embeddings, positive, sigma1 ** 2, sigma2 ** 2) / len(graphs)
            
            labels = batch_graph.y.view(-1, ).to(device)

            # Calculate L1 regularization loss (sum of absolute values of all parameters). This L1 regularization is not used
            # When calculating total loss, only classification loss (classify_loss) and mutual information loss (mi_loss) are used
            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += torch.sum(torch.abs(param))

            classify_loss = criterion(subgraph_output, labels)
            loss = classify_loss + mi_loss * args.mi_weight

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                
                # Add gradient clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # torch.nn.utils.clip_grad_norm_(SG_model.parameters(), max_norm=1.0)
                
                optimizer.step()

            loss = loss.detach().cpu().numpy()

            loss_accum += loss
            miloss_accum += mi_loss

            pbar.set_description(f'epoch: {epoch}')

    print(loss_accum)
    print(len(indices))

    average_loss = float(loss_accum) / len(indices)
    average_miloss = float(miloss_accum) / len(indices)
    print(f"Loss Training: {average_loss}\tMutual Information Loss: {average_miloss}")
    return average_loss, mi_loss


def test(args, model, train_dataset, test_dataset, SG_model, device, criterion=nn.CrossEntropyLoss()):
    """
    A function used to test the trained model that feeds all the testing data into the model once per execution
    """
    model.eval()
    SG_model.eval()

    train_dataset = copy.deepcopy(train_dataset)
    test_dataset = copy.deepcopy(test_dataset)

    num_of_edges_pre = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset)))).edge_index.shape[1]
    for graph in train_dataset:
        subgraph = SG_model(graph)
        graph = subgraph
    num_of_edges_after = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset)))).edge_index.shape[1]

    print(f'edge number (pre/after): ({num_of_edges_pre}/{num_of_edges_after})')

    total_correct_train = 0
    for train_dataset_batch in iter(DataLoader(train_dataset, batch_size=args.batch_size)):
        _, output_train = model(train_dataset_batch)
        _, y_hat_train = torch.max(output_train, dim=1)
        labels_train = train_dataset_batch.y.view(-1).to(device)

        correct = torch.sum(y_hat_train == labels_train)
        total_correct_train += correct

    acc_train = total_correct_train / float(len(train_dataset))
    print(f'train (correct/samples) : ({total_correct_train}/{len(train_dataset)})')

    for graph in test_dataset:
        subgraph = SG_model(graph)
        graph = subgraph

    total_correct_test = 0
    for test_dataset_batch in iter(DataLoader(test_dataset, batch_size=args.batch_size)):
        _, output_test = model(test_dataset_batch)
        _, y_hat_test = torch.max(output_test, dim=1)
        labels_test = test_dataset_batch.y.view(-1, ).to(device)
        test_loss = criterion(output_test, labels_test)
        correct = torch.sum(y_hat_test == labels_test)
        total_correct_test += correct

    acc_test = total_correct_test / float(len(test_dataset))
    print(f'test (correct/samples): ({total_correct_test}/{len(test_dataset)})')

    print("accuracy (train/test): (%f/%f)" % (acc_train, acc_test))

    return acc_train, acc_test, test_loss
