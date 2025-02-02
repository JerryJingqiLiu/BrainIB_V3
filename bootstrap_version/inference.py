import os
import os.path as osp
import argparse
import logging
from datetime import datetime
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data.create_dataset import read_dataset
from src.GNN import GNN
from src.sub_graph_generator import MLP_subgraph

class EnsembleInference:
    """
    Class for ensemble inference using multiple Bootstrap models
    """
    def __init__(
        self,
        model_dir: str,
        device: torch.device,
        num_models: int = 10,
        num_node_features: int = 116,
        num_edge_features: int = 1,
        model_epoch: str = "best"  # New parameter: specify which epoch's model to use
    ):
        self.model_dir = model_dir
        self.device = device
        self.num_models = num_models
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.model_epoch = model_epoch
        
        # Load all models
        self.models, self.sg_models = self._load_models()
        
    def _load_models(self) -> Tuple[List[nn.Module], List[nn.Module]]:
        """
        Load all saved models
        """
        models = []
        sg_models = []
        
        for i in range(self.num_models):
            # Initialize model
            model = GNN(num_of_features=self.num_node_features, device=self.device).to(self.device)
            sg_model = MLP_subgraph(
                node_features_num=self.num_node_features,
                edge_features_num=self.num_edge_features,
                device=self.device
            ).to(self.device)
            
            # Construct model filename based on the specified epoch
            if self.model_epoch == "best":
                model_filename = "best_model.tar"
            else:
                model_filename = f"epoch_{self.model_epoch}.tar"
            
            # Load model state
            model_path = osp.join(self.model_dir, f"bootstrap_{i}", "GNN", model_filename)
            sg_model_path = osp.join(self.model_dir, f"bootstrap_{i}", "subgraph", model_filename)
            
            # Check if the file exists
            if not osp.exists(model_path) or not osp.exists(sg_model_path):
                raise FileNotFoundError(
                    f"Model file does not exist:\n"
                    f"GNN model: {model_path}\n"
                    f"Subgraph model: {sg_model_path}"
                )
            
            model.load_state_dict(torch.load(model_path)["state_dict"])
            sg_model.load_state_dict(torch.load(sg_model_path)["state_dict"])
            
            model.eval()
            sg_model.eval()
            
            models.append(model)
            sg_models.append(sg_model)
            
        return models, sg_models
    
    def predict_single_model(
        self,
        model: nn.Module,
        sg_model: nn.Module,
        data: List[Data]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions using a single model
        
        Returns:
            Predicted classes and probabilities
        """
        batch_size = 32
        all_probs = []
        all_preds = []
        
        # Clone data to avoid modifying the original data
        test_data = [d.clone() for d in data]
        
        # Apply subgraph generator
        for i in range(len(test_data)):
            with torch.no_grad():
                subgraph, _ = sg_model(test_data[i])
                test_data[i] = subgraph
        
        # Batch prediction
        for i in range(0, len(test_data), batch_size):
            batch_data = test_data[i:i + batch_size]
            batch_loader = DataLoader(batch_data, batch_size=len(batch_data))
            batch = next(iter(batch_loader))
            
            with torch.no_grad():
                _, output = model(batch)
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_probs.append(probs)
                all_preds.append(preds)
        
        # Merge results from all batches
        all_probs = torch.cat(all_probs, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        
        return all_preds, all_probs
    
    def predict_ensemble(
        self,
        data: List[Data]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Make ensemble predictions using all models
        
        Returns:
            Ensemble predicted classes, probabilities, and accuracy for each model
        """
        all_model_preds = []
        all_model_probs = []
        model_accuracies = {}
        
        # Get true labels
        true_labels = torch.tensor([d.y.item() for d in data]).to(self.device)
        
        # Make predictions using each model
        for i, (model, sg_model) in enumerate(zip(self.models, self.sg_models)):
            preds, probs = self.predict_single_model(model, sg_model, data)
            all_model_preds.append(preds)
            all_model_probs.append(probs)
            
            # Calculate accuracy for each model
            accuracy = (preds == true_labels).float().mean().item()
            model_accuracies[f"model_{i}_accuracy"] = accuracy
        
        # Stack all predictions
        stacked_preds = torch.stack(all_model_preds)
        stacked_probs = torch.stack(all_model_probs)
        
        # Calculate ensemble predictions (majority voting)
        ensemble_preds = torch.mode(stacked_preds, dim=0).values
        
        # Calculate ensemble probabilities (average)
        ensemble_probs = torch.mean(stacked_probs, dim=0)
        
        # Calculate accuracy of the ensemble model
        ensemble_accuracy = (ensemble_preds == true_labels).float().mean().item()
        model_accuracies["ensemble_accuracy"] = ensemble_accuracy
        
        return ensemble_preds, ensemble_probs, model_accuracies

def main():
    parser = argparse.ArgumentParser(description="Brain Network Analysis Inference")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--num_models",
        type=int,
        default=10,
        help="Number of bootstrap models (default: 10)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_results/",
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--model_epoch",
        type=str,
        default="best",
        help="Specify which epoch's model weights to use. Can be a specific epoch number or 'best' (using the best model)",
    )
    args = parser.parse_args()
    
    # Set output directory
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up logging
    log_filename = osp.join(args.output_dir, f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    dataset = read_dataset()
    
    # Split dataset (keeping the same split as during training)
    np.random.seed(0)  # Use the same random seed as during training
    indices = np.random.permutation(len(dataset))
    split_idx = int(0.7 * len(dataset))  # 70% for training
    inference_indices = indices[split_idx:]  # 30% for inference
    inference_data = [dataset[i] for i in inference_indices]
    
    logger.info(f"Inference dataset size: {len(inference_data)}")
    logger.info(f"Using model weights from: {'best model' if args.model_epoch == 'best' else f'epoch {args.model_epoch}'}")
    
    # Initialize inferencer
    inferencer = EnsembleInference(
        model_dir=args.model_dir,
        device=device,
        num_models=args.num_models,
        model_epoch=args.model_epoch
    )
    
    # Perform inference
    logger.info("Starting inference...")
    ensemble_preds, ensemble_probs, accuracies = inferencer.predict_ensemble(inference_data)
    
    # Record results
    logger.info("\nInference Results:")
    logger.info("=" * 50)
    for model_name, accuracy in accuracies.items():
        logger.info(f"{model_name}: {accuracy:.4f}")
    
    # Save prediction results
    results = {
        "predictions": ensemble_preds.cpu().numpy(),
        "probabilities": ensemble_probs.cpu().numpy(),
        "accuracies": accuracies,
        "model_epoch": args.model_epoch
    }
    
    results_path = osp.join(args.output_dir, f"inference_results_{args.model_epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")
    np.savez(results_path, **results)
    logger.info(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main() 