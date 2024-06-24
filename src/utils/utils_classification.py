
import copy
import json
import logging
import os
from statistics import mean
# from .thread_worker import ThreadWorker
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
import torch_geometric
import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torchtnt.utils.flops import FlopTensorDispatchMode

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")


class UtilsClassification:
    """Class for storing all the utility functions for classification problems.
    The idea is that the functions of this class are reusable for any type
    of model, so that the main programs follow a very similar schema.

    """

    def __init__(
        self,
        problem_name: str,
        dataset_name: str,
        json_output_path: str,
        num_epochs_checkpoint: int = 30,
        num_epochs_log_information: int = 10,
    ):
        """Constructor for the Utils class.

        Args:
            problem_name (str): Name of the problem to solve. Example: Heterogeneous Device Mapping.
            dataset_name (str): Name of the dataset that is used.
            json_output_path (str): Output of the json file where the output is stored.
            num_epochs_checkpoint (int, optional): Number that determines how often a checkpoint is stored. Defaults to 30.
            num_epochs_log_information (int, optional): Number that determines how often training information is logged. Defaults to 10.
        """
        self.processed_data_path = os.path.join("raw_data", problem_name, "processed")
        self.labels_path = os.path.join(
            "raw_data", problem_name, "dataflow", "labels", dataset_name
        )

        self.root_checkpoint_path = os.path.join("checkpoints")
        self.problem_checkpoint_path = os.path.join(
            self.root_checkpoint_path, problem_name
        )
        self.dataset_checkpoint_path = os.path.join(
            self.problem_checkpoint_path, dataset_name
        )
        self.programl_vocab_path = os.path.join(
            "raw_data", problem_name, "vocab", "programl.csv"
        )
        self.json_output_path = json_output_path

        self.num_epochs_checkpoint = num_epochs_checkpoint
        self.num_epochs_log_information = num_epochs_log_information

        self.logger = logging.getLogger("info_logger")
        self.logger.setLevel(logging.INFO)

    def save_model_checkpoint(
        self,
        model: torch.nn.Module,
        num_epoch: int,
        f1_score: float,
        loss: Union[float, torch.Tensor],
        optimizer: torch.optim,
    ):
        """Function to store a checkpoint of a model at a given
        epoch. It stores the model's state and its current f1
        score, so that the checkpoint with the highest score
        can be retrieved.

        Args:
            model (torch.nn.Module): Model to save.
            num_epoch (int): Training epoch of the checkpoint.
            f1_score (float): Validation F1 score of the epoch,
            loss (Union[float, torch.Tensor]): Validation
                loss of the epoch.
            optimizer (torch.optim): Optimizer to save.
        """

        filename = str(num_epoch) + ".Checkpoint.pt"
        checkpoint_filename = os.path.join(self.dataset_checkpoint_path, filename)

        dict_state = {
            "model": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "f1_score": f1_score,
        }

        torch.save(dict_state, checkpoint_filename)

    def load_model_checkpoint(
        self,
    ) -> Dict:
        """Function to load the model checkpoint with the highest
        validation F1 score.

        Raises:
            Exception: If there are no checkpoints to select.

        Returns:
            Dict: Checkpoint of the model.
        """

        if not os.listdir(self.dataset_checkpoint_path):
            raise Exception("There are no checkpoints to select.")

        best_f1 = -1
        best_checkpoint = None

        for filename in os.listdir(self.dataset_checkpoint_path):
            checkpoint_filename = os.path.join(self.dataset_checkpoint_path, filename)
            checkpoint = torch.load(checkpoint_filename)

            temp_f1 = checkpoint["f1_score"]
            if temp_f1 > best_f1:
                best_f1 = temp_f1
                best_checkpoint = checkpoint

        return best_checkpoint

    def log_training_information(
        self,
        num_epoch: int,
        train_loss: float,
        train_metrics: dict,
        val_loss: float,
        val_metrics: dict,
    ):
        """Auxiliar function to log all the training information.

        Args:
            num_epoch (int): Training epoch.
            train_loss (float): Training loss of the epoch.
            train_metrics (dict): Dictionary that contains the
                training accuracy metrics of the epoch.
            val_loss (float): Validation loss of the epoch.
            val_metrics (dict): Dictionary taht contains the
                validation accuracy metrics of the epoch.
        """

        f1_train = train_metrics["f1"]
        f1_val = val_metrics["f1"]

        out_str = "Epoch Number: " + str(num_epoch)
        out_str += (
            ", Train loss: "
            + "{:0.3f}".format(train_loss)
            + ", Train F1 score: "
            + "{:0.3f}".format(f1_train)
        )
        out_str += (
            ", Validation loss: "
            + "{:0.3f}".format(val_loss)
            + ", Validation F1 score: "
            + "{:0.3f}".format(f1_val)
        )

        tqdm.tqdm.write(out_str)

    def print_results(self, metrics: dict, is_test: bool):
        """Function to print all the final metrics
        of the training or testing process.

        Args:
            metrics (dict): Metrics to print.
            is_test (bool): Whether the metrics correspond
                to validation or test process.
        """

        # Get the final results in strings
        header_str = "\n\nVALIDATION RESULTS:"
        underline_str = "===================\n"

        if is_test:
            header_str = "\n\nTEST RESULTS:"
            underline_str = "===============\n"

        f1_str = "F1 score: " + str(metrics["f1"]) + "\n"
        acc_str = "Accuracy: " + str(metrics["accuracy"]) + "\n"
        error_str = "Error score: " + str(metrics["error"]) + "\n"
        precision_str = "Precision score: " + str(metrics["precision"]) + "\n"
        recall_str = "Recall score: " + str(metrics["recall"]) + "\n"

        # Print the strings
        print(header_str)
        print(underline_str)
        print(f1_str)
        print(acc_str)
        print(error_str)
        print(precision_str)
        print(recall_str)

    def save_results(
        self,
        results_dict: dict,
    ):
        """Auxiliar function to store the final metrics
        in json format.

        Args:
            metrics (dict): Metrics to store.

        """
        json_metrics = json.dumps(results_dict, indent=2)

        with open(self.json_output_path, "w") as outfile:
            outfile.write(json_metrics)

        self.logger.info(
            "Output file with the metrics created at: " + str(self.json_output_path)
        )

    def get_programl_vocabulary_length(self) -> int:
        """Function to get the number of tokens
        in the ProGraML dictionary.

        Returns:
            int: Number of tokens in the vocabulary.
        """
        vocab_df = pd.read_csv(self.programl_vocab_path, sep="\t")
        return len(vocab_df)

    def empty_path(self, path: str):
        """Auxiliar function to remove all the files
        of the given path.

        Args:
            path (str): Path to empty.
        """
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)

            if os.path.isfile(file_path):
                os.remove(file_path)

            else:
                print(
                    "Found file with name: {}, that cannot be removed.".format(filename)
                )

    def initialize_checkpoints_directory(self):
        """Function for initializing the directories where the
        training checkpoints are stored. This function is called
        at the beginning of the training process.

        """
        if not os.path.exists(self.root_checkpoint_path):
            os.mkdir(self.root_checkpoint_path)

        if not os.path.exists(self.problem_checkpoint_path):
            os.mkdir(self.problem_checkpoint_path)

        if not os.path.exists(self.dataset_checkpoint_path):
            os.mkdir(self.dataset_checkpoint_path)

        else:
            self.empty_path(self.dataset_checkpoint_path)

    @torch.no_grad()
    def profile_model(
        self, model: torch.nn.Module, test_loader: DataLoader, device: str
    ) -> Tuple[float, int, float]:
        """Function for profiling a deep learning model.
        It computes the number of parameters of the model,
        the needed flops of its forward pass and the execution
        time of its forward pass.

        Args:
            model (torch.nn.Module): Model to profile.
            test_loader (DataLoader): Test dataloader.
            device (str): Device to execute the testing.

        Returns:
            Tuple[float, int, float]: Flops of the forward pass,
                number of parameters of the model and execution
                time of the forward pass.
        """

        # Put the model in eval()
        # mode not to compute gradients
        model.eval()

        model_size = torch_geometric.profile.count_parameters(model)

        # Create CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Measure the execution time
        exec_times = []
        for data in test_loader:
            data = data.to(device)

            start_event.record()
            out = model(data)
            end_event.record()
            torch.cuda.synchronize()

            exec_time_ms = start_event.elapsed_time(end_event)
            exec_times.append(exec_time_ms)

        final_exec_time = mean(exec_times)

        # Measure the flops of the forward pass
        with FlopTensorDispatchMode(model) as ftdm:
            for data in test_loader:
                data = data.to(device)

                out = model(data)

                flops_forward = copy.deepcopy(ftdm.flop_counts)
                ftdm.reset()

                break

        total_flops = 0
        keys = ["mp1", "pooling", "readout"]

        for key in keys:
            layer_flops = flops_forward.get(key, None)
            if layer_flops is not None:
                for item, value in layer_flops.items():
                    if key == "mp1":
                        value = value * 6
                    total_flops += value

        return total_flops, model_size, final_exec_time

    def compute_dataset_distribution(self, dataset: InMemoryDataset) -> Dict[int, int]:
        """Function for computing the labels distribution of the given
        load balancing dataset.

        Args:
            dataset (InMemoryDataset): Input dataset.

        Returns:
            Dict[int, int]: Label and number of graphs with that label.
        """

        labels = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

        for graph in dataset:
            label = graph["y"].item()

            labels[label] += 1

        return labels

    def train(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        criterion: torch.nn,
        optimizer: torch.optim,
        device: str,
    ) -> Tuple[Dict[str, int], float]:
        """Function for training one epoch the input model.

        Args:
            model (torch.nn.Module): Model to train.
            train_loader (DataLoader): DataLoader with the training
                data.
            criterion (torch.nn): Loss function of the model.
            optimizer (torch.optim): Optimizer of the model's
                parameters.
            device (str): Device to execute the training.

        Returns:
            Tuple[Dict[str, int], float]:
                Dict[str, float] : Dictionary containing accuracy metrics of the
                    training epoch.
                float: Loss of the training epoch.
        """
        model.train()

        total_loss = 0
        total_preds = torch.tensor([])
        total_y = torch.tensor([])

        for data in train_loader:  # Iterate in batches over the training dataset.
            data = data.to(device)
            out_logits = model(data)  # Perform a single forward pass.

            loss = criterion(out_logits, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

            # Concat the results of the minibatch
            # to compute metrics afterwards
            total_loss += loss.item()
            out_labels = torch.argmax(out_logits, dim=1)

            # Send data back to cpu
            out_labels_cpu = out_labels.cpu()
            data_y_cpu = data.y.cpu()

            total_preds = torch.cat((total_preds, out_labels_cpu))
            total_y = torch.cat((total_y, data_y_cpu))

            del out_logits
            del loss

        # Compute accuracy metrics
        accuracy = accuracy_score(total_y, total_preds)
        error = 1 - accuracy
        f1 = f1_score(total_y, total_preds, average="weighted")
        precision = precision_score(
            total_y, total_preds, zero_division=0.0, average="weighted"
        )
        recall = recall_score(
            total_y, total_preds, zero_division=0.0, average="weighted"
        )

        # Store the metrics in a dictionary
        metrics = {}
        metrics["accuracy"] = accuracy
        metrics["error"] = error
        metrics["f1"] = f1
        metrics["precision"] = precision
        metrics["recall"] = recall

        return metrics, total_loss

    def validation(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        criterion: torch.nn,
        device: str,
    ) -> Tuple[Dict[str, int], float]:
        """Function for testing the training model against
        the validation dataset.

        Args:
            model (torch.nn.Module): Model to train.
            val_loader (DataLoader): DataLoader with the
                validation data.
            criterion (torch.nn): Loss function of the
                model.
            device (str): Device to execute the training.

        Returns:
            Tuple[Dict[str, int], float]:
                Dict[str, float] : Dictionary containing accuracy metrics of the
                    validation epoch.
                float: Loss of the validation epoch.
        """
        # Put the model in eval()
        # mode not to compute gradients
        model.eval()

        total_loss = 0
        total_preds = torch.tensor([])
        total_y = torch.tensor([])

        # Iterate in batches over
        # validation dataset
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y)

                # Concat the results of the minibatch
                # to compute metrics afterwards
                total_loss += loss.item()
                out_labels = torch.argmax(out, dim=1)

                # Send data back to cpu
                out_labels_cpu = out_labels.cpu()
                data_y_cpu = data.y.cpu()

                total_preds = torch.cat((total_preds, out_labels_cpu))
                total_y = torch.cat((total_y, data_y_cpu))

        # Compute accuracy metrics
        accuracy = accuracy_score(total_y, total_preds)
        error = 1 - accuracy
        f1 = f1_score(total_y, total_preds, average="weighted")
        precision = precision_score(
            total_y, total_preds, zero_division=0.0, average="weighted"
        )
        recall = recall_score(
            total_y, total_preds, zero_division=0.0, average="weighted"
        )

        # Store the metrics in a dictionary
        metrics = {}
        metrics["accuracy"] = accuracy
        metrics["error"] = error
        metrics["f1"] = f1
        metrics["precision"] = precision
        metrics["recall"] = recall

        return metrics, total_loss

    def test(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        criterion: torch.nn,
        device: str,
    ) -> Tuple[Dict[str, int], float]:
        """Function for testing the training model against
        the test dataset.

        Args:
            model (torch.nn.Module): Model to test.
            test_loader (DataLoader): DataLoader with the
                test data.
            criterion (torch.nn): Loss function of the
                model.
            device (str): Device to execute the training.

        Returns:
            Tuple[Dict[str, int], float]:
                Dict[str, float] : Dictionary containing accuracy metrics of the
                    test epoch.
                float: Loss of the test epoch.
        """

        # Put the model in eval()
        # mode not to compute gradients
        model.eval()

        total_loss = 0
        total_preds = torch.tensor([])
        total_y = torch.tensor([])

        # Iterate in batches over
        # test dataset
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data)

                loss = criterion(out, data.y)
                out_labels = torch.argmax(out, dim=1)

                # Send data back to cpu
                out_labels_cpu = out_labels.cpu()
                data_y_cpu = data.y.cpu()

                # Concat the results of the minibatch
                # to compute metrics afterwards
                total_loss += loss.item()
                total_preds = torch.cat((total_preds, out_labels_cpu))
                total_y = torch.cat((total_y, data_y_cpu))
                del data

        # Compute accuracy metrics
        accuracy = accuracy_score(total_y, total_preds)
        error = 1 - accuracy
        f1 = f1_score(total_y, total_preds, average="weighted")
        precision = precision_score(
            total_y, total_preds, zero_division=0.0, average="weighted"
        )
        recall = recall_score(
            total_y, total_preds, zero_division=0.0, average="weighted"
        )

        # Store the metrics in a dictionary
        metrics = {}
        metrics["accuracy"] = accuracy
        metrics["error"] = error
        metrics["f1"] = f1
        metrics["precision"] = precision
        metrics["recall"] = recall

        return metrics, total_loss
