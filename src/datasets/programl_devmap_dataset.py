import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import programl as pg
import torch
from programl.proto import program_graph_pb2
from programl.util.py import pbutil
from torch_geometric.data import HeteroData, InMemoryDataset, download_url


class ProgramlDevmapDataset(InMemoryDataset):
    """
    Class to load the heterogeneous device mapping device described
    at Cummins et al. (2017).

    Note that this class is self contained, meaning that it downloads,
    processes and saves the data automatically.

    For a better understanding of the code, take a look at:
    https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
    """

    def __init__(
        self,
        root: str,
        dataset_name: str = "nvidia",
        transform: Callable = None,
        pre_transform: Callable = None,
        pre_filter: Callable = None,
    ):
        """Constructor of the device mapping dataset.

        Args:
            root (str): Root path where the dataset will be stored.
            dataset_name (str, optional): Name of the device mapping experiment. Defaults to "nvidia".
            transform (Callable, optional): A function that transforms the graph of the dataset.
                The data object will be transformed before every access. Defaults to None.
            pre_transform (Callable, optional): A function that transforms the graph of the dataset.
                The data object will be transformed before being stored to disk. Defaults to None.
            pre_filter (Callable, optional): A function that takes in a graph and returns a boolean value,
                indicating whether the data object should be included in the final dataset. Defaults to None.
        """
        super().__init__(root, transform, pre_transform, pre_filter)

        # Load the dataset of the given device
        if dataset_name == "nvidia":
            self.data, self.slices = torch.load(self.processed_paths[0])

        else:
            self.data, self.slices = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self) -> List[str]:
        """
        Files that are downloaded to ./root/raw directory.

        Files:
        - amd.csv: Csv file containing labels, auxiliar data and kernel codes of the AMD GPU dataset.
        - nvidia.csv: Csv file containing labels, auxiliar data and kernel codes of the NVIDIA GPU dataset.
        - devmap_data.zip: Zip file containing kernel codes and their respective LLVM codes.
        - vocab_20.06.01.tar.bz2: Tar.bz2 file containing vocabularies of the inst2vec, cdfg and ProGraML
            papers.
        """
        return ["amd.csv", "nvidia.csv", "devmap_data.zip", "vocab_20.06.01.tar.bz2"]

    @property
    def processed_file_names(self) -> List[str]:
        """
        Files that are stored at ./root/processed directory after all the processing is done.

        Files:
        - data_nvidia.pt: File containing all the HeteroGraphs corresponding to the Nvidia dataset.
        - data_amd.pt: File containing all the HeteroGraphs corresponding to the Amd dataset.
        """
        return ["data_nvidia.pt", "data_amd.pt"]

    def download(self):
        """
        Method to download self.raw_file_names into self.raw_dir.

        """
        amd_url = "http://raw.githubusercontent.com/ChrisCummins/phd/65643fa5ad6769ce4678535cd2f9f37b6a467c45/datasets/opencl/device_mapping/amd.csv"
        nvidia_url = "http://raw.githubusercontent.com/ChrisCummins/phd/65643fa5ad6769ce4678535cd2f9f37b6a467c45/datasets/opencl/device_mapping/nvidia.csv"
        opencl_ir_url = "https://www.dropbox.com/s/j5ck80fsbuebf5g/devmap_data.zip?dl=1"
        vocab_url = (
            "https://zenodo.org/record/4247595/files/vocab_20.06.01.tar.bz2?download"
        )

        # Download to `self.raw_dir`.
        download_url(amd_url, self.raw_dir)
        download_url(nvidia_url, self.raw_dir)
        download_url(opencl_ir_url, self.raw_dir)
        download_url(vocab_url, self.raw_dir)

    def __extract_vocab_targz(self):
        """
        Auxiliar method to extract the vocabulary .tar.bz2 at self.root directory.

        """
        # Path of the vocabulary .tar.bz2 file
        vocab_tarfile_path = os.path.join(self.raw_dir, "vocab_20.06.01.tar.bz2")

        if not os.path.exists(vocab_tarfile_path):
            raise Exception("Vocabulary .tar.bz2 file does not exist!")

        # Extract the file
        tar = tarfile.open(vocab_tarfile_path)
        tar.extractall(self.root)
        tar.close()

    def __extract_llvm_ir_code(self):
        """
        Function to extract devmap_data.zip at self.root/temp_llvm_ir temporal
        directory. This temporal directory is removed at the end of the process.

        """
        # Create the needed paths
        llvm_ir_zip_path = os.path.join(self.raw_dir, "devmap_data.zip")
        temp_dest_path = os.path.join(self.root, "temp_llvm_ir")

        if not os.path.exists(llvm_ir_zip_path):
            raise Exception("Llvm IR .zip file does not exist!")

        # Create a temporal file to extract all the contents
        if not os.path.exists(temp_dest_path):
            os.mkdir(temp_dest_path)

        try:
            # Extract contents to temporal file
            with zipfile.ZipFile(llvm_ir_zip_path, "r") as zip:
                zip.extractall(temp_dest_path)

        except Exception as error:
            print(error)

    def __name2ncc_path(self, name: str, src_dir: str, extension: str):
        """Resolve a NCC data archive path from a kernel name."""

        path = os.path.join(src_dir, name) + extension
        if os.path.isfile(path):
            return path

        # Some of the benchmark sources are dataset dependent. This is reflected by
        # the dataset name being concatenated to the path.
        name_components = name.split("-")

        new_name1 = "-".join(name_components[:-1])
        path1 = os.path.join(src_dir, new_name1) + extension
        if os.path.isfile(path1):
            return path1

        new_name2 = "-".join(name_components[:-1]) + "_" + name_components[-1]
        path2 = os.path.join(src_dir, new_name2) + extension

        if os.path.isfile(path2):
            return path2

        raise Exception(f"No OpenCL source found for {name}")

    def __load_programl_vocab(self) -> Dict[str, int]:
        """Auxiliar function to load the ProGraML vocabulary.

        Returns:
            Dict[str, int]: Python dictionary where the keys are the
                vocabulary token and the values are their
                respectives indexes.

                vocab[token] = token_index.
        """

        vocab_file_path = os.path.join(self.root, "vocab", "programl.csv")

        # Read the vocabulary csv into a dataframe
        vocab_df = pd.read_csv(vocab_file_path, sep="\t")
        vocab = {}

        # Fill the vocabulary dictionary
        for idx, row in vocab_df.iterrows():
            token = row["text"]
            vocab[token] = idx

        return vocab

    def __load_dataframe(self, df_path: str) -> pd.DataFrame:
        """
        Auxiliar function to load the auxiliar data and labels from
        self.raw_dir/amd.csv or self.raw_dir/nvidia.csv.

        Args:
            df_path (str): Path of the csv file to be loaded.

        Returns:
            pd.DataFrame: DataFrame containing auxiliar data and labels
                of the kernels of the given csv file.
        """

        df = pd.read_csv(df_path)

        # Get the kernel names
        names = [
            f"{benchmark}-{dataset}"
            for benchmark, dataset in df[["benchmark", "dataset"]].values
        ]

        return pd.DataFrame(
            {
                "name": names,
                "transfer_bytes": df["transfer"],
                "transfer_bytes_log1p": np.log1p(df["transfer"]),
                "wgsize": df["wgsize"],
                "wgsize_log1p": np.log1p(df["wgsize"]),
                "label": df["runtime_gpu"] < df["runtime_cpu"],
            }
        )

    def __copy_files(
        self, input_path: str, dest_path: str, extension: str, df: pd.DataFrame
    ):
        """
        Auxiliar function for copying files from the temporal directort to
        dest_path.

        Args:
            input_path (str): Input path to read the files from.
            dest_path (str): Path to copy all the files to.
            extension (str): Extension of the files to copy.
            df (pd.DataFrame): Dataframe containing the names of the files.
        """

        if not os.path.exists(dest_path):
            os.mkdir(dest_path)

        for name in df["name"].values:
            try:
                src = self.__name2ncc_path(name, input_path, extension)
                dst = os.path.join(dest_path, f"{name}{extension}")

                shutil.copyfile(src, dst)
            except Exception:
                pass

    def __create_dataframe_graphs(
        self, df: pd.DataFrame, input_path: str, output_path: str
    ):
        """
        Auxiliar function to create and store all the ProGraML graphs.

        Args:
            df (pd.DataFrame): Dataframe containing code names, auxiliar features
                and labels of the input programs.
            input_path (str): Path containing all the LLVM codes.
            output_path (str): Path where all the ProGraML are stored.
        """

        # Create the output_path in case it
        # does not exist
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        for _, row in df.iterrows():
            # Name of the LLVM file
            llvm_path = os.path.join(input_path, row["name"] + ".ll")

            # Read the LLVM code
            with open(llvm_path, "r") as llvm_file:
                llvm_code = llvm_file.read()

            # Create the ProGraML graph
            graph = pg.from_llvm_ir(llvm_code)

            # Add features to the graph (label + dynamic data)
            graph.features.feature["devmap_label"].int64_list.value[:] = [row["label"]]
            graph.features.feature["wgsize"].int64_list.value[:] = [row["wgsize"]]
            graph.features.feature["transfer_bytes"].int64_list.value[:] = [
                row["transfer_bytes"]
            ]
            graph.features.feature["wgsize_log1p"].float_list.value[:] = [
                row["wgsize_log1p"]
            ]
            graph.features.feature["transfer_bytes_log1p"].float_list.value[:] = [
                row["transfer_bytes_log1p"]
            ]

            # Store the graph at output_path
            graph_path = os.path.join(output_path, row["name"] + ".ProgramGraph.pb")
            graph_path = Path(graph_path)
            pbutil.ToFile(graph, graph_path, exist_ok=True)

    def __programl_to_heterograph(
        self, programl_graph_path: str, vocabulary: Dict[str, int]
    ) -> HeteroData:
        """
        Function to create a torch_geometric.data.HeteroData object from a ProGraML
        graph.

        Args:
            programl_graph_path (str): Path of a ProGraML graph.
            vocabulary (Dict[str, int]): Vocabulary dictionary of the ProGraML graphs.

        Returns:
            HeteroData: The graph in HeteroData format.
        """

        # Load the ProGraML graph.
        programl_graph_path = Path(programl_graph_path)
        programl_graph = pbutil.FromFile(
            programl_graph_path, program_graph_pb2.ProgramGraph()
        )

        # 3 lists, one per edge type
        # (control, data and call edges)
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

        # Obtain all the auxiliar features
        label = programl_graph.features.feature["devmap_label"].int64_list.value[0]
        wgsize = programl_graph.features.feature["wgsize"].int64_list.value[0]
        transfer_bytes = programl_graph.features.feature[
            "transfer_bytes"
        ].int64_list.value[0]
        wgsize_log1p = programl_graph.features.feature["wgsize_log1p"].float_list.value[
            0
        ]
        transfer_bytes_log1p = programl_graph.features.feature[
            "transfer_bytes_log1p"
        ].float_list.value[0]

        # Pass from list to tensor
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
        hetero_graph["nodes", "data", "nodes"].edge_index = (
            adjacencies[1].t().contiguous()
        )
        hetero_graph["nodes", "call", "nodes"].edge_index = (
            adjacencies[2].t().contiguous()
        )

        # Add the edge positions
        hetero_graph["nodes", "control", "nodes"].edge_attr = edge_positions[0]
        hetero_graph["nodes", "data", "nodes"].edge_attr = edge_positions[1]
        hetero_graph["nodes", "call", "nodes"].edge_attr = edge_positions[2]

        # Add the graph label
        hetero_graph.y = label

        # Add the auxiliar features
        hetero_graph["wgsize"] = wgsize
        hetero_graph["transfer_bytes"] = transfer_bytes
        hetero_graph["wgsize_log1p"] = wgsize_log1p
        hetero_graph["transfer_bytes_log1p"] = transfer_bytes_log1p

        return hetero_graph

    def __create_heterograph_dataset(
        self, input_path: str, vocabulary: Dict[str, int]
    ) -> List[HeteroData]:
        """
        Function to create a list of torch_geometric.data.HeteroData. It reads
        the ProGraML graphs one by one, it creates an HeteroData for each of them
        and stores them in a list.

        Args:
            input_path (str): Input path where all the ProGraML graphs are stored.
            vocabulary (Dict[str, int]): Vocabulary dictionary for the ProGraML tokens.

        Returns:
            List[HeteroData]: List containing all the HeteroData objects.

        """
        graph_list = []

        # For each ProGraML graph create an instance
        # of HeteroData and append it to the list
        for graph_name in os.listdir(input_path):
            graph_path = os.path.join(input_path, graph_name)
            hetero_graph = self.__programl_to_heterograph(graph_path, vocabulary)
            graph_list.append(hetero_graph)

        return graph_list

    def process(self):
        """
        Function for making all the processing of the dataset.

        """

        # Get all the auxiliar paths
        amd_csv_path = os.path.join(self.raw_dir, "amd.csv")
        nvidia_csv_path = os.path.join(self.raw_dir, "nvidia.csv")

        data_path = os.path.join(self.root, "original_data")
        ir_path = os.path.join(data_path, "ir")
        src_path = os.path.join(data_path, "src")
        nvidia_graph_path = os.path.join(data_path, "nvidia_graphs")
        amd_graph_path = os.path.join(data_path, "amd_graphs")

        # Extract the vocabulary and
        # the LLVM code
        self.__extract_vocab_targz()
        self.__extract_llvm_ir_code()

        if not os.path.exists(data_path):
            os.mkdir(data_path)

        temp_path = os.path.join(self.root, "temp_llvm_ir")
        temp_ir_path = os.path.join(self.root, "temp_llvm_ir", "kernels_ir")
        temp_src_path = os.path.join(self.root, "temp_llvm_ir", "kernels_cl")

        # Load vocabulary for programl tokens
        programl_vocab = self.__load_programl_vocab()

        # Open dataframes
        amd_df = self.__load_dataframe(amd_csv_path)
        nvdia_df = self.__load_dataframe(nvidia_csv_path)

        # Copy .cl and .ll files
        self.__copy_files(temp_ir_path, ir_path, ".ll", amd_df)
        self.__copy_files(temp_src_path, src_path, ".cl", amd_df)

        # Remove temporal directory
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

        # Convert .ll to programl
        self.__create_dataframe_graphs(amd_df, ir_path, amd_graph_path)
        self.__create_dataframe_graphs(nvdia_df, ir_path, nvidia_graph_path)

        # Convert programl to HeteroData
        nvidia_heterographs = self.__create_heterograph_dataset(
            nvidia_graph_path, programl_vocab
        )
        amd_heterographs = self.__create_heterograph_dataset(
            amd_graph_path, programl_vocab
        )

        # Save to .pt
        data_nvidia, slices_nvidia = self.collate(nvidia_heterographs)
        torch.save((data_nvidia, slices_nvidia), self.processed_paths[0])

        data_amd, slices_amd = self.collate(amd_heterographs)
        torch.save((data_amd, slices_amd), self.processed_paths[1])

        print("PROCESSING DONE!")
