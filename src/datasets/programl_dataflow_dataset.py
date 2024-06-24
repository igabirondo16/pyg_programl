import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd
import torch
import tqdm
from programl.proto import program_graph_pb2, util_pb2
from programl.util.py import pbutil
from torch_geometric.data import (Dataset, HeteroData, InMemoryDataset,
                                  download_url, extract_tar)

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")


class ProgramlDataflowDataset(InMemoryDataset):
    """Auxiliary class for downloading and pre-processing the DeepDataFlow dataset
    presented in "ProGraML: Graph-based Deep Learning for Program Optimization and Analysis" by Cummins
    et al. (2020).

    This class is self-contained, it downloads, processes and saves the data automatically.

    This class is valid for the five dataflow experiments.
    """

    def __init__(
        self,
        root: str,
        dataset_name: str = "liveness",
        max_dataflow_steps: int = 30,
        data_type: str = "train",
        debug: bool = False,
        transform: Callable = None,
        pre_transform: Callable = None,
        pre_filter: Callable = None,
    ):
        """Constructor of the dataflow dataset.

        Args:
            root (str): Root path where the dataset will be stored.
            dataset_name (str, optional): Name of the dataflow experiment. Defaults to "liveness".
            max_dataflow_steps (int, optional): Maximum number of dataflow steps allowed.
                All the graphs that have a more dataflow steps than this value will not be stored.
                Defaults to 30.
            data_type (str, optional): Type of graph to load (training, validation or test).
                Defaults to "train".
            debug (bool, optional): Load a small fraction of the dataset for making tests.
                Defaults to False.
            transform (Callable, optional): A function that transforms the graph of the dataset.
                The data object will be transformed before every access. Defaults to None.
            pre_transform (Callable, optional): A function that transforms the graph of the dataset.
                The data object will be transformed before being stored to disk. Defaults to None.
            pre_filter (Callable, optional): A function that takes in a graph and returns a boolean value,
                indicating whether the data object should be included in the final dataset. Defaults to None.
        """
        self.dataset_name = dataset_name
        self.max_dataflow_steps = max_dataflow_steps
        self.debug = debug
        self.num_graphs_debug = 1000

        if self.debug:
            self.dataset_name = "debug"

        self.labels_path = os.path.join(root, "dataflow", "labels", dataset_name)

        self.train_graphs_path = os.path.join(root, "dataflow", "train")
        self.val_graphs_path = os.path.join(root, "dataflow", "val")
        self.test_graphs_path = os.path.join(root, "dataflow", "test")

        self.graph_paths = [
            self.train_graphs_path,
            self.val_graphs_path,
            self.test_graphs_path,
        ]

        self.logger = logging.getLogger("info_logger")
        self.logger.setLevel(logging.INFO)

        # Start the processing of the data
        super().__init__(root, transform, pre_transform, pre_filter)

        pre_filter_path = os.path.join(self.processed_dir, "pre_filter.pt")
        pre_transform_path = os.path.join(self.processed_dir, "pre_transform.pt")

        if os.path.exists(pre_filter_path):
            os.remove(pre_filter_path)

        if os.path.exists(pre_transform_path):
            os.remove(pre_transform_path)

        # Load the type of graphs
        if data_type == "train":
            self.logger.info(f"Opening {self.processed_paths[0]}.")
            self.data, self.slices = torch.load(self.processed_paths[0])

        elif data_type == "val":
            self.logger.info(f"Opening {self.processed_paths[1]}.")
            self.data, self.slices = torch.load(self.processed_paths[1])

        else:
            self.logger.info(f"Opening {self.processed_paths[2]}.")
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self) -> List[str]:
        """File names of the raw files to download, when starting the processing
        of the data.

        Returns:
            List[str]: Names of the raw file names.
        """
        return [
            "graphs_20.06.01.tar.bz2",
            "labels_reachability_20.06.01.tar.bz2",
            "labels_domtree_20.06.01.tar.bz2",
            "labels_datadep_20.06.01.tar.bz2",
            "labels_liveness_20.06.01.tar.bz2",
            "labels_subexpressions_20.06.01.tar.bz2",
            "vocab_20.06.01.tar.bz2",
        ]

    @property
    def processed_file_names(self) -> List[str]:
        """File names of the .pt files that store the graphs.
        Notice that these can be accessed through self.processed_paths.

        Returns:
            List[str]: Names of the processed .pt files.
        """
        train_processed_graphs = os.path.join(
            self.dataset_name, f"{self.dataset_name}_train.pt"
        )
        val_processed_graphs = os.path.join(
            self.dataset_name, f"{self.dataset_name}_val.pt"
        )
        test_processed_graphs = os.path.join(
            self.dataset_name, f"{self.dataset_name}_test.pt"
        )

        return [train_processed_graphs, val_processed_graphs, test_processed_graphs]

    def download(self):
        """
        Function for downloading and extracting the files defined in self.raw_file_names
        """

        graphs_url = (
            "https://zenodo.org/record/4247595/files/graphs_20.06.01.tar.bz2?download=1"
        )
        vocab_url = (
            "https://zenodo.org/record/4247595/files/vocab_20.06.01.tar.bz2?download=1"
        )

        reachability_labels_url = "https://zenodo.org/record/4247595/files/labels_reachability_20.06.01.tar.bz2?download=1"
        domtree_labels_url = "https://zenodo.org/record/4247595/files/labels_domtree_20.06.01.tar.bz2?download=1"
        datadep_labels_url = "https://zenodo.org/record/4247595/files/labels_datadep_20.06.01.tar.bz2?download=1"
        liveness_labels_url = "https://zenodo.org/record/4247595/files/labels_liveness_20.06.01.tar.bz2?download=1"
        subexpressions_labels_url = "https://zenodo.org/record/4247595/files/labels_subexpressions_20.06.01.tar.bz2?download=1"

        download_url(graphs_url, self.raw_dir)
        download_url(vocab_url, self.raw_dir)

        download_url(reachability_labels_url, self.raw_dir)
        download_url(domtree_labels_url, self.raw_dir)
        download_url(datadep_labels_url, self.raw_dir)
        download_url(liveness_labels_url, self.raw_dir)
        download_url(subexpressions_labels_url, self.raw_dir)

        # Extract the files
        for raw_file_name in self.raw_file_names:
            raw_file_name_path = os.path.join(self.raw_dir, raw_file_name)
            extract_tar(raw_file_name_path, self.root, mode="r:bz2")

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

    def __load_graph_configurations(
        self,
        graph_features_path: Path,
        hetero_data_graph: HeteroData,
        node_list: list,
    ) -> List[HeteroData]:
        """Function for loading the different configurations of the same graph.

        Args:
            graph_features_path (Path): Path of the file that stores the
                features of the graph.
            hetero_data_graph (HeteroData): Base hetero data graph, this does not
                store the labels nor the selector ids.
            node_list (list): List of all the nodes in the graph.

        Returns:
            List[HeteroData]: List with the variations of the same graph.
        """

        graph_dict = hetero_data_graph.to_dict()

        # Read the features of the graph
        graph_features = pbutil.FromFile(
            graph_features_path, util_pb2.ProgramGraphFeaturesList()
        )

        temp_graphs = []
        for graph_conf_idx, graph_conf_features in enumerate(graph_features.graph):
            step_count_feature = graph_conf_features.features.feature[
                "data_flow_step_count"
            ].int64_list.value

            if len(step_count_feature):
                step_count = step_count_feature[0]

            else:
                step_count = 0

            # Filter out the graph if the dataflow steps exceed the limit
            if step_count > self.max_dataflow_steps:
                continue

            # Read the selector ids
            selector_ids = [
                graph_conf_features.node_features.feature_list["data_flow_root_node"]
                .feature[n]
                .int64_list.value[0]
                for n in node_list
            ]

            # Read the labels
            node_labels = [
                graph_conf_features.node_features.feature_list["data_flow_value"]
                .feature[n]
                .int64_list.value[0]
                for n in node_list
            ]

            # Create a new hetero_data
            temp_graph = HeteroData().from_dict(graph_dict)

            selector_ids = torch.tensor(selector_ids)
            node_labels = torch.tensor(node_labels)

            # Add the labels and the selector ids
            temp_graph["nodes"]["selector_ids"] = selector_ids
            temp_graph.y = node_labels

            temp_graphs.append(temp_graph)

        return temp_graphs

    def __programl_to_heterograph(
        self,
        programl_graph_path: str,
        vocabulary: dict,
        programl_graph_original_name: str,
        dataset_idx: int = 0,
    ) -> Tuple[HeteroData, List[int]]:
        """Function to create a torch_geometric.data.HeteroData object from a ProGraML
        graph.

        Args:
            programl_graph_path (str): Path of a ProGraML graph.
            vocabulary (dict): Vocabulary dictionary of the ProGraML graphs.
            programl_graph_original_name (str): Name of the file of the graph.
            dataset_idx (int, optional): Idx of the dataset (0 ==> train, 1 ==> Validation, 2 ==> Test).
                Defaults to 0.

        Returns:
            Tuple[HeteroData, List[int]]: The graph in HeteroData format and the list of the nodes of the graph.
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

        # Get the number of nodes in the graph
        node_list = list(range(len(programl_graph.node)))

        # Pass from list to tensor
        adjacencies = [torch.tensor(adj_flow_type) for adj_flow_type in adjacencies]
        edge_positions = [
            torch.tensor(edge_pos_flow_type) for edge_pos_flow_type in edge_positions
        ]
        vocab_ids = torch.tensor(vocab_ids)

        # Create the graph structure
        hetero_graph = HeteroData()

        # Store original name for later processing
        hetero_graph["original_name"] = programl_graph_original_name

        # Store the dataset the graph belongs to
        hetero_graph["dataset_idx"] = dataset_idx

        # Vocabulary index of each node
        hetero_graph["nodes"].x = vocab_ids

        # Root node for each experiment
        hetero_graph["nodes"]["selector_ids"] = -1

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
        hetero_graph.y = -1

        return hetero_graph, node_list

    def process(self):
        """Function that processes and stores the graphs in HeteroData format."""

        # Load the vocabulary
        vocab = self.__load_programl_vocab()

        processed_graphs_experiment_path = os.path.join(
            self.processed_dir, self.dataset_name
        )

        if not os.path.exists(processed_graphs_experiment_path):
            os.mkdir(processed_graphs_experiment_path)

        train_processed_graphs = os.path.join(
            self.processed_dir, self.dataset_name, f"{self.dataset_name}_train.pt"
        )
        val_processed_graphs = os.path.join(
            self.processed_dir, self.dataset_name, f"{self.dataset_name}_val.pt"
        )
        test_processed_graphs = os.path.join(
            self.processed_dir, self.dataset_name, f"{self.dataset_name}_test.pt"
        )

        for idx, raw_graph_path in enumerate(self.graph_paths):
            self.logger.info(f"Processing graphs of path: {raw_graph_path}")

            num_graphs = 0
            processed_graphs = []
            for graph_name in tqdm.tqdm(
                os.listdir(raw_graph_path), desc="Processing graphs..."
            ):
                graph_path = os.path.join(self.root, "dataflow", "graphs", graph_name)

                splitted_graph_name = graph_name.split(".")
                # Remove .ProgramGraph.pb from original name
                # This name is used for adding the missing features
                graph_raw_name = ".".join(
                    [
                        splitted_graph_name[0],
                        splitted_graph_name[1],
                        splitted_graph_name[2],
                    ]
                )

                graph_features_name = ".".join(
                    [graph_raw_name, "ProgramGraphFeaturesList.pb"]
                )
                graph_features_path = os.path.join(
                    self.labels_path, graph_features_name
                )
                graph_features_path = Path(graph_features_path)

                if not graph_features_path.is_file():
                    continue

                # Load a graph and its configurations
                try:
                    hetero_data_graph, node_list = self.__programl_to_heterograph(
                        graph_path, vocab, graph_raw_name, idx
                    )
                    hetero_data_configurations = self.__load_graph_configurations(
                        graph_features_path, hetero_data_graph, node_list
                    )

                except Exception:
                    continue

                # Store them in a list
                num_graphs += len(hetero_data_configurations)
                processed_graphs.extend(hetero_data_configurations)

                if self.debug and num_graphs > self.num_graphs_debug:
                    break

            self.logger.info(f"{raw_graph_path} has {num_graphs} graphs.")
            print()

            # Store all the graphs in their corresponding file
            if idx == 0:
                torch.save(self.collate(processed_graphs), train_processed_graphs)
                self.logger.info(
                    f"Train graphs successfully saved at: {train_processed_graphs}"
                )

            elif idx == 1:
                torch.save(self.collate(processed_graphs), val_processed_graphs)
                self.logger.info(
                    f"Validation graphs successfully saved at: {val_processed_graphs}"
                )

            else:
                torch.save(self.collate(processed_graphs), test_processed_graphs)
                self.logger.info(
                    f"Test graphs successfully saved at: {test_processed_graphs}"
                )
