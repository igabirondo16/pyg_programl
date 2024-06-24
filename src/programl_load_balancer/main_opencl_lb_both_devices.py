#!/usr/bin/env python
import argparse
import datetime
import os
import sys
import time

import numpy as np
import pyopencl as cl

current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

from typing import Tuple

from torch_geometric.data import HeteroData

from datasets.programl_load_balancing_dataset import \
    ProgramlLoadBalancingDataset
from models.programl_load_balancing.PROGRAML_load_balancing.PROGRAML_load_balancing import \
    ProgramlLoadBalancingModel
from utils.decorators import measure_execution_time
from utils.kernel_utils import kernel_to_heterodata
from utils.lb_utils import (get_devices, get_programl_vocabulary_length,
                            launch_task_to_device, read_input_kernel)
from utils.model_utils import (compute_factor, get_checkpoint,
                               train_production_model)

PROGRAML_VOCAB_PATH = "./utils/programl.csv"


def parse_arguments():

    parser = argparse.ArgumentParser(
        prog="Deep Learning based OpenCL Load Balancer",
        description="Python script for testing on the deep learning model as a real OpenCL load balancer.",
        epilog="Implementation done by Iñigo Gabirondo López (University of Zaragoza).",
    )

    parser.add_argument(
        "--input_kernel_path",
        action="store",
        type=str,
        default="./kernels/test.cl",
        help="Path of the OpenCL kernel to execute.",
    )
    parser.add_argument(
        "--execution_device",
        action="store",
        choices=["cpu", "gpu", "both"],
        type=str,
        default="cpu",
        help="Device to execute the kernel.",
    )
    parser.add_argument(
        "--input_size",
        action="store",
        type=int,
        default=0,
        help="Total size in bytes of the input arguments of the OpenCL kernel",
    )
    parser.add_argument(
        "--wgsize",
        action="store",
        type=int,
        default=0,
        help="Work group size that uses the OpenCL kernel",
    )
    parser.add_argument(
        "--discretize_problem",
        action="store_true",
        default=False,
        help="Whether to consider the problem as regression or classification.",
    )
    parser.add_argument(
        "--predict_balance",
        action="store_true",
        default=False,
        help="Whether to only predict the balancing and skip the execution of the kernel",
    )

    args = parser.parse_args()
    return args


def initialize_context(device: cl.Device) -> Tuple[cl.Context, cl.CommandQueue]:
    """Function that initializes the OpenCL context
    of the given device,

    Args:
        device (cl.Device): Device that will run the OpenCL kernel.

    Returns:
        Tuple[cl.Context, cl.CommandQueue]: OpenCL context and Command Queue
            of the device.
    """

    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(ctx)

    return ctx, queue


def initialize_model_dataset(
    discretize_problem: bool,
) -> Tuple[ProgramlLoadBalancingDataset, ProgramlLoadBalancingModel]:
    """Function for loading the training dataset and the load
    balancing model.
    The hyper-parameters of the model are hard-coded from main_programl_cross_validation.py

    Args:
        discretize_problem (bool): Whether to consider the problem as classification.

    Returns:
        Tuple[ProgramlLoadBalancingDataset, ProgramlLoadBalancingModel]: The load balancing
            training dataset and model.
    """
    dataset = ProgramlLoadBalancingDataset(
        root="./../raw_data/load_balancing",
        discretize_problem=discretize_problem,
    )

    model = ProgramlLoadBalancingModel(
        vocab_size=get_programl_vocabulary_length() + 1,
        hidden_size=64,
        edge_type_count=3,
        use_edge_bias=True,
        dropout_rate_linear_layer=0.3,
        dropout_rate_update_layer=0.3,
        num_classes=5,
        output_dropout=0.3,
        aux_features_dim=2,
        graph_x_layer_size=192,
        use_forward_and_backward_edges=True,
        discretize_problem=discretize_problem,
        pooling_method="skips",
        extended_readout=True
    )

    return dataset, model


def divide_input_data(
    model: ProgramlLoadBalancingModel,
    heterodata_graph: HeteroData,
    discretize_problem: bool,
):
    """Function for predicting the workload division of the input kernel.

    Args:
        model (ProgramlLoadBalancingModel): Load balancing model.
        heterodata_graph (HeteroData): Input kernel in ProGraML format.
        discretize_problem (bool): Whether to consider the problem as classification.
    """
    cpu_factor = compute_factor(model, heterodata_graph, discretize_problem)
    print(f"Predicted workload division (CPU - GPU): {cpu_factor} - {1-cpu_factor}")

    print(f"CPU percentage: {cpu_factor}")
    print(f"GPU percentage: {1-cpu_factor}")


def main():
    """Script for computing the workload division.
    It is under construction
    """
    args = parse_arguments()

    input_kernel_path = args.input_kernel_path
    execution_device = args.execution_device
    discretize_problem = args.discretize_problem
    predict_balance = args.predict_balance
    input_size = args.input_size
    wgsize = args.wgsize

    dataset, model = initialize_model_dataset(discretize_problem)

    model_checkpoint = get_checkpoint(discretize_problem)
    if model_checkpoint is None:
        train_production_model(model, dataset, discretize_problem)

        model_checkpoint = get_checkpoint(discretize_problem)

    model.load_state_dict(model_checkpoint["model"])

    start_time = time.monotonic()

    platforms = cl.get_platforms()
    devices = get_devices(platforms)

    device_gpu = devices[0]
    device_cpu = devices[1]

    ctx_cpu, queue_cpu = initialize_context(device_cpu)
    ctx_gpu, queue_gpu = initialize_context(device_gpu)

    input_kernel = read_input_kernel(input_kernel_path)

    if input_kernel is None:
        print(f"The given kernel path ({input_kernel_path}) does not exist. Exiting.")
        return -1

    # Convert the kernel to ProGraML graph.
    heterodata_graph = kernel_to_heterodata(input_kernel_path, PROGRAML_VOCAB_PATH)
    print(heterodata_graph)

    # This has been set because of the device's characteristics
    wgsize_log1p = np.log1p(wgsize)
    dsize_log1p = np.log1p(input_size)

    heterodata_graph["wgsize_log1p"] = wgsize_log1p
    heterodata_graph["transfer_bytes_log1p"] = dsize_log1p

    divide_input_data(
        model, heterodata_graph, discretize_problem
    )


if __name__ == "__main__":
    main()
