import os
import os.path as osp
import copy
import logging
from typing import List, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import numpy as np
from sklearn.model_selection import train_test_split

from .GNN import GNN
from .sub_graph_generator import MLP_subgraph
from .utils import train, test

class BootstrapTrainer:
    """
    A class to handle the bootstrap training process for the brain network model.
    """
    def __init__(
        self,
        args,
        dataset: List[Data],
        device: torch.device,
        logger: logging.Logger,
        num_node_features: int = 116,
        num_edge_features: int = 1,
        num_models: int = 10
    ):
        """
        Initialize the bootstrap trainer.
        
        Args:
            args: Arguments containing training parameters
            dataset: List of graph data
            device: Device to run the models on
            logger: Logger instance
            num_node_features: Number of node features
            num_edge_features: Number of edge features
            num_models: Number of bootstrap models to train
        """
        self.args = args
        self.dataset = dataset
        self.device = device
        self.logger = logger
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.num_models = num_models
        
        # Split dataset into train (70%) and inference (30%) sets
        self.train_data, self.inference_data = self._split_dataset()
        
        # Initialize lists to store models and their performance
        self.models = []
        self.sg_models = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.inference_accuracies = []
        
    def _split_dataset(self) -> Tuple[List[Data], List[Data]]:
        """
        Split the dataset into training and inference sets.
        
        Returns:
            Tuple containing training and inference datasets
        """
        # Get labels for stratification
        labels = [graph.y.item() for graph in self.dataset]
        
        # Split dataset while preserving class distribution
        train_idx, inference_idx = train_test_split(
            range(len(self.dataset)),
            test_size=0.3,
            random_state=self.args.seed,
            stratify=labels
        )
        
        train_data = [self.dataset[i] for i in train_idx]
        inference_data = [self.dataset[i] for i in inference_idx]
        
        return train_data, inference_data
    
    def _bootstrap_sample(self, dataset: List[Data]) -> Tuple[List[Data], List[Data]]:
        """
        Create a bootstrap sample from the dataset.
        
        Args:
            dataset: List of graph data to sample from
            
        Returns:
            Tuple containing bootstrap training and validation datasets
        """
        n_samples = len(dataset)
        # Random sampling with replacement
        train_idx = np.random.choice(n_samples, size=int(0.7 * n_samples), replace=True)
        # Get validation indices (samples not in bootstrap training set)
        val_idx = list(set(range(n_samples)) - set(train_idx))
        
        bootstrap_train = [dataset[i] for i in train_idx]
        bootstrap_val = [dataset[i] for i in val_idx]
        
        return bootstrap_train, bootstrap_val
    
    def train_single_model(self, model_idx: int) -> Tuple[float, float, float]:
        """
        Train a single model using bootstrap sampling.
        
        Args:
            model_idx: Index of the current model
            
        Returns:
            Tuple containing training, validation, and inference accuracies
        """
        # Create bootstrap sample
        bootstrap_train, bootstrap_val = self._bootstrap_sample(self.train_data)
        
        # Initialize models
        model = GNN(num_of_features=self.num_node_features, device=self.device).to(self.device)
        sg_model = MLP_subgraph(
            node_features_num=self.num_node_features,
            edge_features_num=self.num_edge_features,
            device=self.device
        ).to(self.device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam([
            {"params": model.parameters(), "lr": self.args.model_learning_rate},
            {"params": sg_model.parameters(), "lr": self.args.SGmodel_learning_rate}
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        best_val_acc = 0.0
        best_train_acc = 0.0
        best_model_state = None
        best_sg_model_state = None
        
        for epoch in range(1, self.args.epochs + 1):
            # Train the model
            avg_loss, mi_loss = train(
                self.args, model, bootstrap_train, optimizer, epoch, sg_model, self.device
            )
            
            # Test on bootstrap training and validation sets
            train_acc, val_acc, _ = test(
                self.args, model, bootstrap_train, bootstrap_val, sg_model, self.device
            )
            
            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_train_acc = train_acc
                best_model_state = copy.deepcopy(model.state_dict())
                best_sg_model_state = copy.deepcopy(sg_model.state_dict())
            
            self.logger.info(
                f"Model {model_idx}, Epoch {epoch}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}"
            )
            
            scheduler.step()
        
        # Load best model states
        model.load_state_dict(best_model_state)
        sg_model.load_state_dict(best_sg_model_state)
        
        # Test on inference set
        _, inference_acc, _ = test(
            self.args, model, bootstrap_train, self.inference_data, sg_model, self.device
        )
        
        # Save models
        self._save_model(model, sg_model, model_idx)
        
        return best_train_acc, best_val_acc, inference_acc
    
    def _save_model(self, model: nn.Module, sg_model: nn.Module, model_idx: int):
        """
        Save the trained models.
        
        Args:
            model: Trained GNN model
            sg_model: Trained subgraph generator model
            model_idx: Index of the current model
        """
        # Create directory for this bootstrap model
        base_savedir = osp.join(self.args.save_dir, f"bootstrap_{model_idx}")
        if not osp.exists(base_savedir):
            os.makedirs(base_savedir)
        
        # Save GNN model
        model_savedir = osp.join(base_savedir, "GNN")
        if not osp.exists(model_savedir):
            os.makedirs(model_savedir)
        model_savename = osp.join(model_savedir, "best_model.tar")
        torch.save({"state_dict": model.state_dict()}, model_savename)
        
        # Save subgraph model
        subgraph_savedir = osp.join(base_savedir, "subgraph")
        if not osp.exists(subgraph_savedir):
            os.makedirs(subgraph_savedir)
        subgraph_savename = osp.join(subgraph_savedir, "best_model.tar")
        torch.save({"state_dict": sg_model.state_dict()}, subgraph_savename)
        
        self.models.append(model)
        self.sg_models.append(sg_model)
    
    def train_all_models(self):
        """
        Train all bootstrap models and evaluate their performance.
        """
        for i in range(self.num_models):
            self.logger.info(f"\nTraining Bootstrap Model {i+1}/{self.num_models}")
            train_acc, val_acc, inference_acc = self.train_single_model(i)
            
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.inference_accuracies.append(inference_acc)
            
            self.logger.info(
                f"Bootstrap Model {i+1} Results:\n"
                f"Training Accuracy: {train_acc:.4f}\n"
                f"Validation Accuracy: {val_acc:.4f}\n"
                f"Inference Accuracy: {inference_acc:.4f}"
            )
        
        # Log final results
        self.logger.info("\nFinal Results Summary:")
        self.logger.info("=" * 50)
        self.logger.info(f"Number of Bootstrap Models: {self.num_models}")
        self.logger.info(f"Training Accuracies: {self.train_accuracies}")
        self.logger.info(f"Mean Training Accuracy: {np.mean(self.train_accuracies):.4f} ± {np.std(self.train_accuracies):.4f}")
        self.logger.info(f"Validation Accuracies: {self.val_accuracies}")
        self.logger.info(f"Mean Validation Accuracy: {np.mean(self.val_accuracies):.4f} ± {np.std(self.val_accuracies):.4f}")
        self.logger.info(f"Inference Accuracies: {self.inference_accuracies}")
        self.logger.info(f"Mean Inference Accuracy: {np.mean(self.inference_accuracies):.4f} ± {np.std(self.inference_accuracies):.4f}") 