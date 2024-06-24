#!/usr/bin/env python
import argparse
import datetime
import os
import time
from typing import Tuple

import numpy as np
import pyopencl as cl

from utils.kernel_utils import kernel_to_heterodata
from utils.lb_utils import (get_devices, launch_task_to_device,
                            read_input_kernel)

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
        help="Path of the kernel to execute.",
    )
    parser.add_argument(
        "--execution_device",
        action="store",
        choices=["cpu", "gpu", "both"],
        type=str,
        default="cpu",
        help="Device to execute the kernel.",
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


def main():
    """Example script for running a simple vector summation.
    It sends the total computation to a single device
    """
    args = parse_arguments()

    input_kernel_path = args.input_kernel_path
    execution_device = args.execution_device

    print(f"Execution device: {execution_device}")

    # Initialize input data
    a_np = np.random.rand(50000000).astype(np.float32)
    b_np = np.random.rand(50000000).astype(np.float32)

    start_time = time.time()

    platforms = cl.get_platforms()
    devices = get_devices(platforms)

    device_gpu = devices[0]
    device_cpu = devices[1]

    device = device_cpu if execution_device == "cpu" else device_gpu

    print("Device:")
    print(device)

    # Initialize the context and the kernel
    ctx, queue = initialize_context(device)
    input_kernel = read_input_kernel(input_kernel_path)

    if input_kernel is None:
        print(f"The given kernel path ({input_kernel_path}) does not exist. Exiting.")
        return -1

    # This is just for debug reasons
    heterodata_graph = kernel_to_heterodata(input_kernel_path, PROGRAML_VOCAB_PATH)
    print(heterodata_graph)

    # Send the computation and get the result
    res = launch_task_to_device(ctx, queue, input_kernel, a_np=a_np, b_np=b_np)

    end_time = time.time()

    print("RESULTS:")
    print(res - (a_np + b_np))
    print(np.linalg.norm(res - (a_np + b_np)))
    assert np.allclose(res, a_np + b_np)

    print(f"Total execution time: {(end_time - start_time) * 1000} (ms)")


if __name__ == "__main__":
    main()
