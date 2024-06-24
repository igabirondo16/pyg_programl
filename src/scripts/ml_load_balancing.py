import argparse
import os
import time
from statistics import mean
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score
)
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

CSV_PATH = "./../raw_data/load_balancing/raw"


def parse_arguments():
    """
    Function to parse the input arguments.

    """

    parser = argparse.ArgumentParser(
        prog="Machine Learning methods for Load Balancing",
        description="Python script for training/testing classical machine learning methods for load balancing.",
        epilog="Script done by Iñigo Gabirondo López (University of Zaragoza).",
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
        "--discretize_problem",
        action="store_true",
        default=False,
        help="Whether to consider the problem as regression or classification.",
    )
    parser.add_argument(
        "--seed", action="store", type=int, default=8, help="Seed for experiments."
    )
    parser.add_argument(
        "--profile_model",
        action="store_true",
        default=False,
        help="Get the model size and the forward pass flops of the model.",
    )

    args = parser.parse_args()
    return args


def load_csv_dataframe(dataset_name: str) -> pd.DataFrame:
    """Function for reading the heterogeneous device mapping dataset in
    csv format.

    Args:
        dataset_name (str): Name of the csv file.

    Returns:
        pd.DataFrame: The csv dataset in DataFrame format.
    """

    dataset_name = ".".join((dataset_name, "csv"))
    dataset_path = os.path.join(CSV_PATH, dataset_name)

    df = pd.read_csv(dataset_path)

    return df


def get_dataset_labels(df_dataset: pd.DataFrame, discretize_problem: bool) -> np.array:
    """Function for computing the labels of the dataset.

    Args:
        df_dataset (pd.DataFrame): Dataset information in DataFrame format.
        discretize_problem (bool): Whether to consiuder the problem as classification
            or regression.

    Returns:
        np.array: Labels of the dataset. If discretize_problem, the possible labels are
            [0, 1, 2, 3, 4]. Otherwise, it is a continuous value from 0 to 100.
    """
    runtime_cpu = df_dataset["runtime_cpu"].values
    runtime_gpu = df_dataset["runtime_gpu"].values

    # Compute regression value
    factor = runtime_gpu / runtime_cpu
    cpu_percentage = (factor / (1 + factor)).round(decimals=2)  # Round to 2 decimals
    cpu_percentage *= 100.0  # Convert ratio to percentage

    if not discretize_problem:
        return cpu_percentage

    nvidia_labels_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    aux_labels = np.array([0.0, 25.0, 50.0, 75.0, 100.0])

    # Compute classification labels
    labels = []
    for value in cpu_percentage:
        aux = np.full((5), value)
        label = np.abs(aux_labels - aux).argmin()
        nvidia_labels_count[label] += 1
        labels.append(label)

    print("Label distribution:")
    print(nvidia_labels_count)
    labels = np.array(labels)
    return labels


def extract_dataset_features(
    df_dataset: pd.DataFrame, discretize_problem: bool
) -> Tuple[np.array, np.array]:
    """Function for extracting the dataset features and labels that will be used
    for training the machine learning models.

    Args:
        df_dataset (pd.DataFrame): Dataset information in DataFrame format.
        discretize_problem (bool):  Whether to consiuder the problem as classification
            or regression.

    Returns:
        Tuple[np.array, np.array]: Dataset features and labels.
    """
    train_features = np.array(
        [
            (
                df_dataset["transfer"].values
                / (df_dataset["comp"].values + df_dataset["mem"].values)
            ),  # F1
            (df_dataset["coalesced"].values / df_dataset["mem"].values),  # F2
            (
                (df_dataset["localmem"].values / df_dataset["mem"].values)
                * df_dataset["wgsize"].values
            ),  # F3
            (df_dataset["comp"].values / df_dataset["mem"].values),  # F4
        ]
    ).T

    labels = get_dataset_labels(df_dataset, discretize_problem)

    return train_features, labels


def main():
    args = parse_arguments()

    dataset_name = args.dataset_name
    discretize_problem = args.discretize_problem
    seed = args.seed
    profile_model = args.profile_model

    df_dataset = load_csv_dataframe(dataset_name=dataset_name)

    # Load the dataset
    features, labels = extract_dataset_features(df_dataset, discretize_problem)
    kf = KFold(n_splits=10, random_state=seed, shuffle=True)

    if discretize_problem:
        global_f1 = []
        global_acc = []
        global_recall = []
        global_precision = []

    else:
        global_rmse = []
        global_mae = []
        global_mse = []

    if profile_model:
        exec_times = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(features)):

        # Choose the model
        if discretize_problem:
            model = DecisionTreeClassifier(
                random_state=seed,
                splitter="best",
                criterion="entropy",
                max_depth=5,
                min_samples_leaf=5,
            )

        else:
            model = LinearRegression(n_jobs=6)

        train_features = features[train_idx]
        train_labels = labels[train_idx]

        test_features = features[test_idx]
        test_labels = labels[test_idx]

        # Train the model
        model = model.fit(train_features, train_labels)

        if profile_model:
            start_time = time.time()

        # Test the model
        inference_test_labels = model.predict(test_features)

        if profile_model:
            end_time = time.time()
            exec_time = (end_time - start_time) * 1000
            exec_times.append(exec_time)

        # Compute accuracy metrics of the test results
        if discretize_problem:
            accuracy = accuracy_score(test_labels, inference_test_labels)
            f1 = f1_score(test_labels, inference_test_labels, average="weighted")
            precision = precision_score(
                test_labels,
                inference_test_labels,
                zero_division=0.0,
                average="weighted",
            )
            recall = recall_score(
                test_labels,
                inference_test_labels,
                zero_division=0.0,
                average="weighted",
            )

            print()
            print(f"TEST RESULTS OF FOLD {fold_idx}")
            print("==============================")
            print(
                f"Test F1: {f1}, Test Accuracy: {accuracy}, Test precision: {precision}, Test recall: {recall}"
            )

            global_acc.append(accuracy)
            global_f1.append(f1)
            global_precision.append(precision)
            global_recall.append(recall)

        else:
            mae = mean_absolute_error(test_labels, inference_test_labels)
            mse = mean_squared_error(test_labels, inference_test_labels)
            rmse = np.sqrt(mse)

            print()
            print(f"TEST RESULTS OF FOLD {fold_idx}")
            print("==============================")
            print(f"Test RMSE: {rmse}, Test MSE: {mse}, Test MAE: {mae}")

            global_mae.append(mae)
            global_mse.append(mse)
            global_rmse.append(rmse)

    model_name = "Decision Tree"
    if not discretize_problem:
        model_name = "Linear Regression"

    # Show final results
    print()
    print(f"FINAL RESULTS, seed = {seed}, model name: {model_name}")
    print("========================================================")

    if profile_model:
        print(f"Mean execution time: {mean(exec_times)} (ms)")

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
