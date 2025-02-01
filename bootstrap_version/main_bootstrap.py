import os
import os.path as osp
import argparse
import logging
from datetime import datetime

import torch
import numpy as np

from data.create_dataset import read_dataset
from src.bootstrap_trainer import BootstrapTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Network Analysis with Bootstrap Training")
    parser.add_argument(
        "--iters_per_epoch",
        type=int,
        default=50,
        help="number of iterations per each epoch (default: 50)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for dataset splitting and training (default: 0)",
    )
    parser.add_argument(
        "--mi_weight",
        type=float,
        default=1e-3,
        help="weight of mutual information loss (default: 0.001)",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=1e-3,
        help="weight of mutual information loss (default: 0.001)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--model_learning_rate",
        type=float,
        default=0.0005,
        help="learning rate of graph model (default: 0.0005)",
    )
    parser.add_argument(
        "--SGmodel_learning_rate",
        type=float,
        default=0.0005,
        help="learning rate of subgraph model (default: 0.0005)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./workdir/",
        help="directory to save models and logs",
    )
    parser.add_argument(
        "--num_bootstrap_models",
        type=int,
        default=10,
        help="number of bootstrap models to train (default: 10)",
    )
    args = parser.parse_args()

    # Set up logging
    log_filename = osp.join(args.save_dir, f'bootstrap_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Configure logging settings
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting bootstrap training process...")
    logger.info(f"Arguments: {args}")

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load dataset
    dataset = read_dataset()
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize bootstrap trainer
    trainer = BootstrapTrainer(
        args=args,
        dataset=dataset,
        device=device,
        logger=logger,
        num_models=args.num_bootstrap_models
    )

    # Train all bootstrap models
    trainer.train_all_models()

    logger.info("Training completed successfully!") 