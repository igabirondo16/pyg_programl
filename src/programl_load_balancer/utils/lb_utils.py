import os

import numpy as np
import pandas as pd
import pyopencl as cl


def get_programl_vocabulary_length() -> int:
    """Function to get the number of tokens
    in the ProGraML dictionary.

    Returns:
        int: Number of tokens in the vocabulary.
    """

    vocab_df = pd.read_csv("./utils/programl.csv", sep="\t")
    return len(vocab_df)


def read_input_kernel(kernel_path: str) -> str:
    """Function for reading the contents of an
    OpenCL kernel.

    Args:
        kernel_path (str): Path of the OpenCL kernel.

    Returns:
        str: OpenCL code.
    """
    if not os.path.exists(kernel_path):
        return None

    with open(kernel_path, "r") as file:
        file_contents = file.read()

    return file_contents


def get_devices(platforms: list) -> list:
    """Function for getting the OpenCL devices
    from a list of available platforms.

    Args:
        platforms (list): OpenCL platforms available
            in the system.

    Returns:
        list: List of OpenCL devices.
    """

    devices = []
    for idx, platform in enumerate(platforms):

        # Device 0 ==> GPU
        # Device 1 ==> CPU
        # This is chosen because of the architecture
        # of our server
        if idx < 2:
            for device in platform.get_devices():
                devices.append(device)

    return devices


def launch_task_to_device(ctx, queue, input_kernel, **kwargs):
    """
    Function that sends an OpenCL kernel that sums two vectors.
    """
    a_np = kwargs.get("a_np")
    b_np = kwargs.get("b_np")
    both_devices = kwargs.get("both_devices")

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

    prg = cl.Program(ctx, input_kernel).build()

    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
    knl = prg.sum  # Use this Kernel object for repeated calls
    # None ==> Workgroup size
    knl(queue, a_np.shape, None, a_g, b_g, res_g)

    if both_devices:
        return queue, res_g

    res_np = np.empty_like(a_np)
    cl.enqueue_copy(queue, res_np, res_g)

    return res_np
