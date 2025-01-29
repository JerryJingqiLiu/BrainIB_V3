import os
import os.path as osp
import argparse

import torch
import torch.nn.functional as F

import numpy as np

from torch_geometric.loader import DataLoader

from src.GNN import GNN
from src.sub_graph_generator import MLP_subgraph
from src.utils import separate_data, train, test

from data.create_dataset import read_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABIDE")
    parser.add_argument(
        "--fold",
        type=int,
        default=-1,
        help="Specify the fold to run (0-9). Default is -1, which means run all folds",
    )
    parser.add_argument(
        "--iters_per_epoch",
        type=int,
        default=50,
        help="number of iterations per each epoch (default: 1)",
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
        help="random seed for splitting the dataset into 10 (default: 0)",
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
        default="/Users/jingqi/Desktop/MICCAI/V3/brainib/SGSIB/workdir/",
        help="directory to save models and logs",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    dataset = read_dataset()

    num_node_features = 116
    num_edge_features = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    num_of_fold = 10
    acc_train_list = torch.zeros((num_of_fold,))
    acc_test_list = torch.zeros((num_of_fold,))


    if args.fold != -1:
        if args.fold < 0 or args.fold >= num_of_fold:
            raise ValueError(f"fold parameter must be between 0 and {num_of_fold-1}")
        fold_range = range(args.fold, args.fold + 1)
    else:
        fold_range = range(num_of_fold)

    for fold_idx in fold_range:
        print(f"fold_idx: {fold_idx}")
        max_acc_train = 0.0
        max_acc_test = 0.0
        best_train_epoch = 0
        best_test_epoch = 0

        train_dataset, test_dataset = separate_data(dataset, args.seed, fold_idx)

        # Instantiate the backbone network
        model = GNN(num_of_features=num_node_features, device=device).to(device)
        # Instantiate the subgraph generator
        SG_model = MLP_subgraph(
            node_features_num=num_node_features,
            edge_features_num=num_edge_features,
            device=device,
        )

        SG_model = SG_model.to(device)

        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters(), "lr": args.model_learning_rate},
                {"params": SG_model.parameters(), "lr": args.SGmodel_learning_rate}
            ]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        for epoch in range(1, args.epochs + 1):
            print(f'Current FOLD: ---{fold_idx}---')
            # Train the model and test it
            avg_loss, mi_loss = train(
                args, model, train_dataset, optimizer, epoch, SG_model, device
            )
            acc_train, acc_test, test_loss = test(
                args, model, train_dataset, test_dataset, SG_model, device
            )

            # print info and save models
            if acc_train > max_acc_train:
                max_acc_train = acc_train
                best_train_epoch = epoch
            if acc_test > max_acc_test:
                max_acc_test = acc_test
                best_test_epoch = epoch
            acc_train_list[fold_idx] = max_acc_train
            acc_test_list[fold_idx] = max_acc_test
            print(
                f"Current epoch {epoch}, best accuracy (train: {max_acc_train:.4f} at epoch {best_train_epoch} / "
                f"test: {max_acc_test:.4f} at epoch {best_test_epoch})"
            )

            # Create file structure using user-specified root directory
            base_savedir = osp.join(args.save_dir, f"fold_{fold_idx}")
            if not osp.exists(base_savedir):
                os.makedirs(base_savedir)

            # Save GNN model
            model_savedir = osp.join(base_savedir, "GNN")
            if not osp.exists(model_savedir):
                os.makedirs(model_savedir)
            model_savename = osp.join(model_savedir, f"epoch_{epoch}.tar")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                },
                model_savename,
            )

            # Save subgraph model
            subgraph_savedir = osp.join(base_savedir, "subgraph")
            if not osp.exists(subgraph_savedir):
                os.makedirs(subgraph_savedir)
            subgraph_savename = osp.join(subgraph_savedir, f"epoch_{epoch}.tar")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": SG_model.state_dict(),
                },
                subgraph_savename,
            )

            # Save training log
            log_filename = osp.join(base_savedir, "training_log.txt")
            log_mode = "w" if not os.path.exists(log_filename) else "a+"
            with open(log_filename, log_mode) as f:
                # Write header if it's a new file
                if log_mode == "w":
                    f.write("Epoch\tAvg_Loss\tTrain_Acc\tTest_Acc\tMI_Loss\n")
                
                # Process different types of losses
                if isinstance(avg_loss, (float, int)):
                    avg_loss_scalar = float(avg_loss)
                elif torch.is_tensor(avg_loss):
                    avg_loss_scalar = float(avg_loss.item())
                else:  # numpy array
                    avg_loss_scalar = float(avg_loss[0][0])

                if isinstance(mi_loss, (float, int)):
                    mi_loss_scalar = float(mi_loss)
                elif torch.is_tensor(mi_loss):
                    mi_loss_scalar = float(mi_loss.item())
                else:  # numpy array
                    mi_loss_scalar = float(mi_loss[0][0])

                # Write data with epoch number
                f.write(f"{epoch}\t{avg_loss_scalar:.6f}\t{acc_train:.6f}\t{acc_test:.6f}\t{mi_loss_scalar:.6f}\n")

            scheduler.step()

            torch.cuda.empty_cache()

    print(100 * "-")
    print("ASD 10-fold validation results: ")
    print("Model Name: V3")
    print(f"train accuracy list: {acc_train_list}")
    print(f"mean = {acc_train_list.mean()}, variance = {acc_train_list.var()}")
    print(f"test accuracy list: {acc_test_list}")
    print(f"mean = {acc_test_list.mean()}, variance = {acc_test_list.var()}")
    print(100 * "-")
