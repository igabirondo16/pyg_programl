import argparse
import sys
from statistics import mean

import seaborn as sns
import torch
import torch_geometric
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.loader import DataLoader

from datasets.programl_devmap_dataset import ProgramlDevmapDataset
from datasets.programl_load_balancing_dataset import \
    ProgramlLoadBalancingDataset

from models.programl_load_balancing.GAT_load_balancing.GAT_load_balancing import \
    GATLoadBalancing
from models.programl_load_balancing.GCN_load_balancing.GCN_load_balancing import \
    GCNLoadBalancing
from models.programl_load_balancing.MLP_load_balancing.MLP_load_balancing import \
    MLPLoadBalancing
from models.programl_load_balancing.PROGRAML_load_balancing.PROGRAML_load_balancing import \
    ProgramlLoadBalancingModel

from utils.utils_classification import UtilsClassification
from utils.utils_regression import UtilsRegression

def parse_arguments():
    """
    Function to parse the input arguments.

    """
    parser = argparse.ArgumentParser(
        prog="ProGraML-devmap-pytorch geometric",
        description="Python script for training/testing ProGraML model for the heterogeneous device mapping problem.",
        epilog="Migration to Pytorch Geometric done by Iñigo Gabirondo López (University of Zaragoza).",
    )

    parser.add_argument(
        "--discretize_problem",
        action="store_true",
        default=False,
        help="Whether to consider the problem as regression or classification.",
    )
    parser.add_argument(
        "--devmap",
        action="store_true",
        default=False,
        help="Perform device mapping training.",
    )
    parser.add_argument(
        "--dataset_name",
        action="store",
        choices=["nvidia", "amd"],
        default="nvidia",
        type=str,
        help="The dataset to be used in the experiment.",
    )
    parser.add_argument(
        "--model_name",
        action="store",
        choices=["gcn", "mlp", "gat", "default"],
        default="default",
        help="Model to train.",
    )
    parser.add_argument(
        "--pooling_method",
        action="store",
        choices=["default", "mean", "diff", "topk", "skips"],
        default="skips",
        help="Pooling method used for graph classification.",
    )
    parser.add_argument(
        "--profile_model",
        action="store_true",
        default=False,
        help="Get the model size and the forward pass flops of the model.",
    )
    parser.add_argument(
        "--seed", action="store", type=int, default=1, help="Seed for experiments."
    )
    parser.add_argument(
        "--extended_readout",
        action="store_true",
        default=False,
        help="Whether to use the extended readout or not.",
    )
    parser.add_argument(
        "--num_epochs",
        action="store",
        type=int,
        default=300,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, default=32, help="Batch size."
    )
    parser.add_argument(
        "--lr",
        action="store",
        type=float,
        default=0.001,
        help="Learning rate for the training process.",
    )
    parser.add_argument(
        "--hidden_size",
        action="store",
        type=int,
        default=64,
        help="Dimension for the embedding vectors.",
    )
    parser.add_argument(
        "--use_forward_and_backward_edges",
        action="store_true",
        default=False,
        help="Add backward edges to the graph.",
    )
    parser.add_argument(
        "--use_edge_bias",
        action="store_true",
        default=True,
        help="Use bias in the linear layers of the Message Passing layer.",
    )
    parser.add_argument(
        "--dropout_rate_linear_layer",
        action="store",
        type=float,
        default=0.3,
        help="Dropout rate for the linear layer of the Message Passing layer.",
    )
    parser.add_argument(
        "--dropout_rate_update_layer",
        action="store",
        type=float,
        default=0.3,
        help="Dropout rate for the update layer of the Message Passing layer.",
    )
    parser.add_argument(
        "--dropout_rate_readout",
        action="store",
        type=float,
        default=0.3,
        help="Dropout rate for linear layer of the readout head.",
    )
    parser.add_argument(
        "--aux_features_dim",
        action="store",
        type=int,
        default=2,
        help="Size of the vector of auxiliar features concatenated to the graph hidden vector.",
    )
    parser.add_argument(
        "--graph_x_layer_size",
        action="store",
        type=int,
        default=192,
        help="Dimension of the input vector to the readout head.",
    )
    parser.add_argument(
        "--output_filename",
        action="store",
        type=str,
        default="./output.json",
        help="Path of the filename to write the training/testing results \
                        in a json file.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    seed = args.seed

    generator = torch.Generator().manual_seed(seed)
    torch_geometric.seed_everything(seed)

    model_name = args.model_name

    print(f"MODEL TRAINING: {model_name}")

    # Load the dataset
    output_filename = args.output_filename
    dataset_name = args.dataset_name
    discretize_problem = args.discretize_problem
    device_mapping = args.devmap

    if device_mapping:
        discretize_problem = True
        problem_name = "devmap"

    else:
        problem_name = "load_balancing"

    if discretize_problem:
        util_funcs = UtilsClassification(
            problem_name=problem_name,
            dataset_name=dataset_name,
            json_output_path=output_filename,
        )

    else:
        util_funcs = UtilsRegression(
            problem_name=problem_name,
            dataset_name=dataset_name,
            json_output_path=output_filename,
        )

    if device_mapping:
        dataset = ProgramlDevmapDataset(
            root="./raw_data/devmap", dataset_name=dataset_name
        )

    else:
        print(
            f"Loading dataset: {dataset_name}, Discretize problem: {discretize_problem}"
        )
        dataset = ProgramlLoadBalancingDataset(
            root="./raw_data/load_balancing",
            dataset_name=dataset_name,
            discretize_problem=discretize_problem,
        )

    # Check the distributions of the datasets
    if not discretize_problem:
        labels = util_funcs.compute_dataset_distribution_regression(dataset)
        print(f"Regression dataset. Labels distribution: {labels}")

    else:
        labels = util_funcs.compute_dataset_distribution(dataset)
        print(f"Classification dataset. Labels distribution: {labels}")

    # Load all the input arguments
    profile_model = args.profile_model
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    use_forward_and_backward_edges = args.use_forward_and_backward_edges
    use_edge_bias = args.use_edge_bias
    dropout_rate_linear_layer = args.dropout_rate_linear_layer
    dropout_rate_update_layer = args.dropout_rate_update_layer
    dropout_rate_readout = args.dropout_rate_readout
    aux_features_dim = args.aux_features_dim
    graph_x_layer_size = args.graph_x_layer_size
    vocab_size = (
        util_funcs.get_programl_vocabulary_length() + 1
    )  # Take into account unknown token
    num_classes = dataset.num_classes
    edge_type_count = len(dataset.num_edge_features.keys())
    pooling_method = args.pooling_method
    extended_readout = args.extended_readout

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if discretize_problem:
        criterion = torch.nn.CrossEntropyLoss()

    else:

        criterion = torch.nn.MSELoss()

    # Load the k-fold cross validation
    k_fold = KFold(n_splits=10, shuffle=True, random_state=seed)

    if profile_model:
        total_flops = []
        total_exec_times = []

    if discretize_problem:
        global_f1 = []
        global_acc = []
        global_recall = []
        global_precision = []

    else:
        global_rmse = []
        global_mae = []
        global_mse = []

    for fold, (train_val_ids, test_ids) in enumerate(k_fold.split(dataset)):
        print()
        print(f"FOLD {fold}")
        print("--------------------------------")

        # Initialize the model
        if model_name == "default":
            model = ProgramlLoadBalancingModel(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                edge_type_count=edge_type_count,
                use_edge_bias=use_edge_bias,
                dropout_rate_linear_layer=dropout_rate_linear_layer,
                dropout_rate_update_layer=dropout_rate_update_layer,
                num_classes=num_classes,
                output_dropout=dropout_rate_readout,
                aux_features_dim=aux_features_dim,
                graph_x_layer_size=graph_x_layer_size,
                use_forward_and_backward_edges=use_forward_and_backward_edges,
                discretize_problem=discretize_problem,
                pooling_method=pooling_method,
                extended_readout=extended_readout,
            )

        elif model_name == "mlp":
            model = MLPLoadBalancing(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_classes=num_classes,
                output_dropout=dropout_rate_readout,
                aux_features_dim=aux_features_dim,
                graph_x_layer_size=graph_x_layer_size,
                discretize_problem=discretize_problem,
            )

        elif model_name == "gat":
            model = GATLoadBalancing(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_classes=num_classes,
                output_dropout=dropout_rate_readout,
                aux_features_dim=aux_features_dim,
                graph_x_layer_size=graph_x_layer_size,
                message_passing_timesteps=6,
                discretize_problem=discretize_problem,
            )

        else:
            model = GCNLoadBalancing(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_classes=num_classes,
                output_dropout=dropout_rate_readout,
                aux_features_dim=aux_features_dim,
                graph_x_layer_size=graph_x_layer_size,
                message_passing_timesteps=6,
                discretize_problem=discretize_problem,
            )

        model.reset_parameters()

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model = model.to(device)

        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=0.1, random_state=seed, shuffle=True
        )

        train_random_sampler = torch.utils.data.SubsetRandomSampler(
            train_ids, generator=generator
        )
        val_random_sampler = torch.utils.data.SubsetRandomSampler(
            val_ids, generator=generator
        )
        test_random_sampler = torch.utils.data.SubsetRandomSampler(
            test_ids, generator=generator
        )

        # Sample train and test
        train_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=train_random_sampler
        )
        val_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=val_random_sampler
        )
        test_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=test_random_sampler
        )

        if profile_model:
            flops, model_size, exec_time = util_funcs.profile_model(
                model, test_loader, device
            )
            print(f"Flops of the model: {flops}")
            print(f"Model size: {model_size}")
            print(f"Mean execution time of the forward pass: {exec_time} (ms)")
            total_flops.append(flops)

            if fold > 0:
                total_exec_times.append(exec_time)
            continue

        best_validation_model = model
        best_validation_epoch = -1

        if discretize_problem:
            best_validation_f1 = 0.0

        else:
            best_validation_rmse = sys.maxsize

        for epoch in range(num_epochs):
            # Train the model
            train_results, train_loss = util_funcs.train(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            # Validate the model
            val_metrics, val_loss = util_funcs.validation(
                model=model, val_loader=val_loader, criterion=criterion, device=device
            )

            # Obtain the metrics
            if discretize_problem:
                f1_train = train_results.get("f1")
                f1_validation = val_metrics.get("f1")

                # Get the model with the best validation f1
                if f1_validation > best_validation_f1:
                    best_validation_epoch = epoch
                    best_validation_model = model
                    best_validation_f1 = f1_validation

                if epoch % 25 == 0:
                    print(
                        f"Epoch {epoch}, Train F1 score: {f1_train}, Validation F1 score: {f1_validation}"
                    )

            else:
                rmse_train = train_results.get("mse")
                rmse_val = val_metrics.get("mse")

                if rmse_val < best_validation_rmse:
                    best_validation_epoch = epoch
                    best_validation_rmse = rmse_val
                    best_validation_model = model

                if epoch % 25 == 0:
                    print(
                        f"Epoch {epoch}, Train MSE: {rmse_train}, Validation MSE: {rmse_val}"
                    )

        # Load the checkpoint with the best validation f1
        print()

        if discretize_problem:
            print(
                f"Picking the model with Validation f1 score: {best_validation_f1} of epoch: {best_validation_epoch}"
            )

        else:
            print(
                f"Picking the model with RMSE: {best_validation_rmse} of epoch: {best_validation_epoch}"
            )

        # Test the model
        test_metrics, test_loss = util_funcs.test(
            model=best_validation_model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        # Store the test results
        if discretize_problem:
            test_f1 = test_metrics.get("f1")
            test_acc = test_metrics.get("accuracy")
            test_precision = test_metrics.get("precision")
            test_recall = test_metrics.get("recall")

            print()
            print(f"TEST RESULTS OF FOLD {fold}")
            print("==============================")
            print(
                f"Test F1: {test_f1}, Test Accuracy: {test_acc}, Test precision: {test_precision}, Test recall: {test_recall}"
            )
            # Append the metrics

            global_f1.append(test_f1)
            global_acc.append(test_acc)
            global_precision.append(test_precision)
            global_recall.append(test_recall)

        else:
            test_rmse = test_metrics.get("rmse")
            test_mae = test_metrics.get("mae")
            test_mse = test_metrics.get("mse")

            print()
            print(f"TEST RESULTS OF FOLD {fold}")
            print("==============================")
            print(f"Test RMSE: {test_rmse}, Test MSE: {test_mse}, Test MAE: {test_mae}")

            global_rmse.append(test_rmse)
            global_mae.append(test_mae)
            global_mse.append(test_mse)

    # Print the metrics
    print()
    if device_mapping:
        print("EXPERIMENT: DEVICE MAPPING")

    else:
        print("EXPERIMENT: LOAD BALANCING")

    print()
    print(f"MODEL: {model_name}")
    print(f"Pooling method: {pooling_method}")
    print(f"Extended readout: {extended_readout}")
    print()

    if profile_model:
        print()
        print(f"Mean model flops: {mean(total_flops)}")
        print(f"Mean exec time: {mean(total_exec_times)} (ms)")
        return

    print(f"FINAL RESULTS, seed = {seed}")
    print("===============")

    if discretize_problem:
        print(f"Mean F1 score: {mean(global_f1)}")
        print(f"Mean accuracy: {mean(global_acc)}")
        print(f"Mean precision: {mean(global_precision)}")
        print(f"Mean recall: {mean(global_recall)}")

    else:
        print(f"Mean RMSE: {mean(global_rmse)}")
        print(f"Mean MAE: {mean(global_mae)}")
        print(f"Mean MSE: {mean(global_mse)}")


if __name__ == "__main__":
    main()
