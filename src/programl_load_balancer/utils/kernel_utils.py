import os
import time
from typing import Dict

import pandas as pd
import programl as pg
import torch
from torch_geometric.data import HeteroData


def kernel_to_heterodata(kernel_path: str, vocab_file_path: str) -> HeteroData:
    """Function that converts an OpenCL kernel to ProGraML
    represention in HeteroData format.
    It compiles the kernel to LLVM, from LLVM it converts
    to ProGraML and from ProGraML to HeteroData.

    Args:
        kernel_path (str): Path of the kernel.
        vocab_file_path (str): Path of the ProGraML vocabulary file.

    Returns:
        HeteroData: Kernel in HeteroData format.
    """

    # Convert from OpenCL to ProGraML
    llvm_code = opencl_to_llvm(kernel_path)
    programl_graph = llvm_to_programl(llvm_code)

    # Convert from ProGraML to HeteroData
    programl_vocab = load_vocabulary(vocab_file_path)
    hetero_graph = programl_to_heterodata(programl_graph, programl_vocab)

    return hetero_graph


def opencl_to_llvm(kernel_path: str) -> str:
    """Function that compiles an OpenCL kernel to LLVM.
    It returns the contents of the LLVM file.

    Args:
        kernel_path (str): Path of the kernel.

    Returns:
        str: Contents of the LLVM file.
    """
    full_path = os.path.split(kernel_path)

    path = full_path[0]
    full_filename = full_path[1]

    filename = full_filename.split(".")[0]
    llvm_filename = ".".join((filename, "ll"))
    llvm_path = os.path.join(path, llvm_filename)

    # Command for compiling to LLVM, execute the command from python
    clang_command = "clang-10 -Xclang -finclude-default-header -emit-llvm -S"
    full_path_command = f"-c {kernel_path}"
    llvm_command = f"-o {llvm_path}"

    command = " ".join((clang_command, full_path_command, llvm_command))

    # Use subprocess instead
    os.system(command)

    # TO DO: CREATE A MORE ELEGANT WAY OF WAITING FOR THE COMPILATION
    while not os.path.exists(llvm_path):
        time.sleep(1)

    # Read the contents of the LLVM file
    with open(llvm_path, "r") as file:
        llvm_contents = file.read()

    print(f"Kernel {kernel_path} succesfully converted to LLVM.")

    return llvm_contents


def llvm_to_programl(llvm_code: str):
    """Function to convert from LLVM to ProGraML.

    Args:
        llvm_code (str): LLVM code of the OpenCL kernel.

    Returns:
        The LLVM code in ProGraML graph format.
    """
    graph = pg.from_llvm_ir(llvm_code)

    return graph


def load_vocabulary(vocab_path: str) -> Dict[str, int]:
    """Auxiliar function to load the ProGraML vocabulary.

    Args:
        vocab_path (str): Path of the ProGraML vocabulary.

    Returns:
        Dict[str, int]: Python dictionary where the keys are the
            vocabulary token and the values are their
            respectives indexes.

            vocab[token] = token_index.
    """
    # Read the vocabulary csv into a dataframe
    vocab_df = pd.read_csv(vocab_path, sep="\t")
    vocab = {}

    # Fill the vocabulary dictionary
    for idx, row in vocab_df.iterrows():
        token = row["text"]
        vocab[token] = idx

    return vocab


def programl_to_heterodata(programl_graph, vocabulary: Dict[str, int]) -> HeteroData:
    """Function to convert a ProGraML graph to HeteroData.
    The current ProGraML version already has this function implemented.

    TO DO:
        - Replace this function with to_pyg() method of the ProGraML
        package.

    Args:
        programl_graph: OpenCL kernel in ProGraML format.
        vocabulary (Dict[str, int]): Python dictionary where the keys are the
            vocabulary token and the values are their
            respectives indexes.

    Returns:
        HeteroData: Kernel in HeteroData format.
    """
    adjacencies = [[], [], []]
    edge_positions = [[], [], []]

    # Create the adjacency lists
    for edge in programl_graph.edge:
        adjacencies[edge.flow].append([edge.source, edge.target])
        edge_positions[edge.flow].append(edge.position)

    # Give a vocabulary index to each node of the graph
    # Notice that in case the text is not found, the
    # "unknown token" index will be given
    vocab_ids = [
        vocabulary.get(node.text, len(vocabulary.keys()))
        for node in programl_graph.node
    ]

    adjacencies = [torch.tensor(adj_flow_type) for adj_flow_type in adjacencies]
    edge_positions = [
        torch.tensor(edge_pos_flow_type) for edge_pos_flow_type in edge_positions
    ]
    vocab_ids = torch.tensor(vocab_ids)

    # Create the graph structure
    hetero_graph = HeteroData()

    # Vocabulary index of each node
    hetero_graph["nodes"].x = vocab_ids

    # Add the adjacency lists
    hetero_graph["nodes", "control", "nodes"].edge_index = (
        adjacencies[0].t().contiguous()
    )
    hetero_graph["nodes", "data", "nodes"].edge_index = adjacencies[1].t().contiguous()
    hetero_graph["nodes", "call", "nodes"].edge_index = adjacencies[2].t().contiguous()

    # Add the edge positions
    hetero_graph["nodes", "control", "nodes"].edge_attr = edge_positions[0]
    hetero_graph["nodes", "data", "nodes"].edge_attr = edge_positions[1]
    hetero_graph["nodes", "call", "nodes"].edge_attr = edge_positions[2]

    return hetero_graph
