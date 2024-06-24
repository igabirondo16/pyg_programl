import argparse
import datetime
import logging
import sys
import time

import torch
import torch_geometric
import tqdm
from torch._dynamo.utils import CompileProfiler
from torch_geometric.loader import DataLoader

from datasets.programl_dataflow_dataset import ProgramlDataflowDataset
from models.programl_dataflow.loss import WeightedCrossEntropyLoss
from models.programl_dataflow.programl_dataflow_model import \
    ProgramlDataflowModel
from utils.utils_classification import UtilsClassification

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")


def parse_arguments():
    """
    Function to parse the input arguments.

    """
    parser = argparse.ArgumentParser(
        prog="ProGraML-dataflow-pytorch geometric",
        description="Python script for training/testing ProGraML model for the dataflow problems.",
        epilog="Migration to Pytorch Geometric done by Iñigo Gabirondo López (University of Zaragoza).",
    )

    parser.add_argument(
        "--dataset_name",
        action="store",
        choices=["datadep", "domtree", "liveness", "reachability", "subexpressions"],
        default="liveness",
        type=str,
        help="The dataset to be used in the experiment.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run in debug mode with a smaller dataset.",
    )
    parser.add_argument(
        "--num_classes",
        action="store",
        type=int,
        default=2,
        help="Number of classes in the dataset.",
    )
    parser.add_argument(
        "--from_checkpoint",
        action="store_true",
        default=False,
        help="Initialize model from the training checkpoint that has the highest \
                        F1 validation score.",
    )
    parser.add_argument(
        "--compile_model", action="store_true", default=False, help="Compile the model."
    )
    parser.add_argument(
        "--max_dataflow_steps",
        action="store",
        type=int,
        default=30,
        help="Maximum number of dataflow steps in which a graph has to be \
                        solved not be discarded.",
    )

    parser.add_argument(
        "--use_forward_and_backward_edges",
        action="store_true",
        default=False,
        help="Add backward edges to the graph.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Skip training process and test the model against the training dataset. \
                        There must be a model checkpoint to load.",
    )
    parser.add_argument(
        "--num_epochs",
        action="store",
        type=int,
        default=2,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, default=32, help="Batch size."
    )
    parser.add_argument(
        "--lr",
        action="store",
        type=float,
        default=0.00025,
        help="Learning rate for the training process.",
    )
    parser.add_argument(
        "--lr_decay_rate",
        action="store",
        type=float,
        default=1.0,
        help="Learning rate decay; multiplicative factor for lr after every epoch.",
    )
    parser.add_argument(
        "--loss_weighting",
        action="store",
        type=float,
        default=0.5,
        help="Weight of the loss contribution for mitigating the class imbalance.",
    )
    parser.add_argument(
        "--hidden_size",
        action="store",
        type=int,
        default=32,
        help="Dimension for the embedding vectors.",
    )
    parser.add_argument(
        "--selector_embedding_size",
        action="store",
        type=int,
        default=2,
        help="Dimension for the selector embedding.",
    )
    parser.add_argument(
        "--selector_embedding_value",
        action="store",
        type=float,
        default=50.0,
        help="Value for the selector embedding.",
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
        default=0.0,
        help="Dropout rate for the linear layer of the Message Passing layer.",
    )
    parser.add_argument(
        "--dropout_rate_update_layer",
        action="store",
        type=float,
        default=0.2,
        help="Dropout rate for the update layer of the Message Passing layer.",
    )
    parser.add_argument(
        "--dropout_rate_readout",
        action="store",
        type=float,
        default=0.2,
        help="Dropout rate for linear layer of the readout head.",
    )
    parser.add_argument(
        "--num_message_passing_layers",
        action="store",
        type=int,
        default=1,
        help="Number of Message Passing layers.",
    )
    parser.add_argument(
        "--message_passing_timesteps",
        action="store",
        type=int,
        default=30,
        help="Number of timesteps that each Message Passing layer does.",
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
    args_dict = vars(args)

    return args, args_dict


def main():
    """
    Main program for training/testing ProGraML dataflow model.

    """

    # Parse input arguments
    args, args_dict = parse_arguments()

    logger = logging.getLogger("info_logger")
    logger.setLevel(logging.INFO)

    results_dict = {}

    output_filename = args.output_filename
    do_test = args.test
    problem_name = "dataflow"
    dataset_name = args.dataset_name
    debug = args.debug

    # Initialize utils
    util_funcs = UtilsClassification(
        problem_name=problem_name,
        dataset_name=dataset_name,
        json_output_path=output_filename,
    )

    time1 = time.time()

    # Load datasets
    if do_test:
        test_dataset = ProgramlDataflowDataset(
            root="./raw_data/dataflow",
            data_type="test",
            dataset_name=dataset_name,
            debug=debug,
        )

    else:
        train_dataset = ProgramlDataflowDataset(
            root="./raw_data/dataflow",
            data_type="train",
            dataset_name=dataset_name,
            debug=debug,
        )
        val_dataset = ProgramlDataflowDataset(
            root="./raw_data/dataflow",
            data_type="val",
            dataset_name=dataset_name,
            debug=debug,
        )

    time2 = time.time()
    dataset_preprocess_time = str(datetime.timedelta(seconds=time2 - time1))

    logger.info(f"Dataset preprocessing done. Elapsed time: {dataset_preprocess_time}")

    # Get all the input arguments
    from_checkpoint = args.from_checkpoint
    compile_model = args.compile_model
    max_dataflow_steps = args.max_dataflow_steps

    lr = args.lr
    num_epochs = args.num_epochs
    gamma = args.lr_decay_rate
    loss_weighting = args.loss_weighting
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    selector_embedding_size = args.selector_embedding_size
    selector_embedding_value = args.selector_embedding_value
    use_edge_bias = args.hidden_size
    use_forward_and_backward_edges = args.use_forward_and_backward_edges
    dropout_rate_linear_layer = args.dropout_rate_linear_layer
    dropout_rate_update_layer = args.dropout_rate_update_layer
    dropout_rate_readout = args.dropout_rate_readout
    num_message_passing_layers = args.num_message_passing_layers
    message_passing_timesteps = args.message_passing_timesteps
    vocab_size = (
        util_funcs.get_programl_vocabulary_length() + 1
    )  # Take into account unknown token

    if not do_test:
        num_classes = train_dataset.num_classes
        edge_type_count = len(train_dataset.num_edge_features.keys())

    else:
        num_classes = test_dataset.num_classes
        edge_type_count = len(test_dataset.num_edge_features.keys())

    epochs = (
        10000,
        20000,
        30000,
        40000,
        50000,
        100000,
        200000,
        300000,
        400000,
        500000,
        600000,
        700000,
        800000,
        900000,
        1000000,
    )

    if debug:
        epochs = [10000]

    num_graphs_eval_test = 10000

    args_dict["epochs"] = epochs
    args_dict["num_graphs_eval_test"] = num_graphs_eval_test

    if not output_filename.endswith(".json"):
        raise Exception("The output path MUST be json format.")

    time3 = time.time()
    filtering_time = str(datetime.timedelta(seconds=time3 - time2))

    logger.info(f"Dataset filtering done. Elapsed time: {filtering_time}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = ProgramlDataflowModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        selector_embedding_size=selector_embedding_size,
        selector_embedding_value=selector_embedding_value,
        edge_type_count=edge_type_count,
        use_edge_bias=use_edge_bias,
        use_forward_and_backward_edges=use_forward_and_backward_edges,
        dropout_rate_linear_layer=dropout_rate_linear_layer,
        dropout_rate_update_layer=dropout_rate_update_layer,
        num_message_passing_layers=num_message_passing_layers,
        message_passing_timesteps=message_passing_timesteps,
        num_classes=num_classes,
        output_dropout=dropout_rate_readout,
        device=device,
        edge_position_max=4096,
    )

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = WeightedCrossEntropyLoss(loss_weighting=loss_weighting, device=device)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma, last_epoch=-1, verbose=True
    )

    # Compilin the model actually does not work!!
    if compile_model:
        prof = CompileProfiler()
        model = torch_geometric.compile(model)
        torch.set_float32_matmul_precision("high")
        torch._dynamo.config.cache_size_limit = 256

    model = model.to(device)

    # Initialize the model's parameters
    if from_checkpoint:
        best_checkpoint = util_funcs.load_model_checkpoint()
        model.load_state_dict(best_checkpoint["model"])

    else:
        model.reset_parameters()

    if do_test:
        global_train_metrics = {}

        # Check the model against the test dataset
        if not from_checkpoint:
            raise Exception("A checkpoint must be loaded in order to test the model.")

        # Load the full test dataset
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=6)

        # Run the test
        test_metrics, test_loss = util_funcs.test(model, test_loader, criterion, device)
        train_metrics = {}

        time4 = time.time()
        training_time = "00:00:00"
        testing_time = str(datetime.timedelta(seconds=time4 - time3))
        total_time = str(datetime.timedelta(seconds=time4 - time1))

        print()
        logger.info(f"Testing finished. Elapsed time: {testing_time}")
        logger.info(
            f"Execution finished, proceeding to save the results. Total execution time: {total_time}"
        )

        util_funcs.print_results(test_metrics, is_test=True)

    else:
        # Train the model
        util_funcs.initialize_checkpoints_directory()

        global_train_metrics = {}

        manual_seed = 13

        # For each epoch, sample the number of graphs and train
        for epoch_idx, epoch_num_grahs in tqdm.tqdm(
            enumerate(epochs), desc="Training...", file=sys.stdout, position=0
        ):

            generator = torch.Generator().manual_seed(manual_seed)

            # Random sample from train dataset
            train_random_sampler = torch.utils.data.RandomSampler(
                train_dataset, num_samples=epoch_num_grahs, generator=generator
            )

            # Random sample for validation dataset
            val_random_sampler = torch.utils.data.RandomSampler(
                val_dataset, num_samples=num_graphs_eval_test, generator=generator
            )

            # Load the sampled training graphs
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_random_sampler,
                num_workers=6,
            )

            # Load the sampled validation graphs
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=val_random_sampler,
                num_workers=6,
            )

            # Run the training
            train_metrics, train_loss = util_funcs.train(
                model, train_dataloader, criterion, optimizer, device
            )

            # Run teh validation
            val_metrics, val_loss = util_funcs.validation(
                model, val_dataloader, criterion, device
            )

            util_funcs.log_training_information(
                epoch_idx, train_loss, train_metrics, val_loss, val_metrics
            )

            global_train_metrics[epoch_idx] = {}
            global_train_metrics[epoch_idx]["train_metrics"] = train_metrics
            global_train_metrics[epoch_idx]["val_metrics"] = val_metrics

            val_f1_score = val_metrics.get("f1", 0.0)

            util_funcs.save_model_checkpoint(
                model, epoch_idx, val_f1_score, val_loss, optimizer
            )

            manual_seed += 1

        test_metrics = {}

        time4 = time.time()
        training_time = str(datetime.timedelta(seconds=time4 - time3))
        testing_time = "00:00:00"
        total_time = str(datetime.timedelta(seconds=time4 - time1))

        print()
        logger.info(f"Training finished. Elapsed time: {training_time}")
        logger.info(
            f"Execution finished, proceeding to save the results. Total execution time: {total_time}"
        )

        # Training is over, print and store the results
        util_funcs.print_results(train_metrics, is_test=False)

    # Store all the metrics
    time_dict = {}
    time_dict["preprocess"] = dataset_preprocess_time
    time_dict["filter"] = filtering_time
    time_dict["training"] = training_time
    time_dict["testing"] = testing_time
    time_dict["total"] = total_time

    results_dict["args"] = args_dict
    results_dict["time"] = time_dict
    results_dict["training_metrics"] = global_train_metrics
    results_dict["testing_metrics"] = test_metrics

    util_funcs.save_results(results_dict)


if __name__ == "__main__":
    main()
