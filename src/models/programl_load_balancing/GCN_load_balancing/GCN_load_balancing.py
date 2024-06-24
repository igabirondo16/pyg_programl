import torch
from torch import nn
from torch_geometric.nn import GCNConv

from .layers.node_embeddings import NodeEmbeddings
from .layers.readout import Readout


class GCNLoadBalancing(nn.Module):
    """Load Balancing model that uses GCN layers instead of the
    ProGraML GNN.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_classes: int,
        output_dropout: float,
        aux_features_dim: int,
        graph_x_layer_size: int,
        message_passing_timesteps: int,
        discretize_problem: bool,
    ):
        """Constructor of the GCNLoadBalancing.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Dimension of the hidden vectors.
            num_classes (int): Number of classes of the dataset.
            output_dropout (float): Probability of using Dropout
                in the readout layer. Notice that if output_dropout == 0.0
                it has no regularization effect.
            aux_features_dim (int): Dimension of the auxiliar features
                vector.
            graph_x_layer_size (int): Dimension of the graph vector in
                the intermediate layers of the feed_forward layer.
            message_passing_timesteps (int): Number of message passing iterations
                to perform.
            discretize_problem (bool): Whether to use the model for regression or
                classification. If discretize_problem, the model works for classification.
        """

        super().__init__()
        self._message_passing_timesteps = message_passing_timesteps

        self.node_embeddings = NodeEmbeddings(
            vocab_size=vocab_size, embedding_size=hidden_size
        )

        self.mp1 = GCNConv(
            in_channels=hidden_size,
            out_channels=hidden_size,
        )

        self.mp2 = GCNConv(
            in_channels=hidden_size,
            out_channels=hidden_size,
        )

        self.readout = Readout(
            num_classes=num_classes,
            hidden_size=hidden_size,
            output_dropout=output_dropout,
            aux_features_dim=aux_features_dim,
            graph_x_layer_size=graph_x_layer_size,
            discretize_problem=discretize_problem,
        )

    def reset_parameters(self):
        self.node_embeddings.reset_parameters()
        self.mp1.reset_parameters()
        self.mp2.reset_parameters()
        self.readout.reset_parameters()

    def forward(self, batch_data):

        # Get the node vocabulary and selector ids
        node_vocab_ids = batch_data["nodes"]["x"]

        # Extract adjacency matrices per edge type
        control_edge_index = batch_data["nodes", "control", "nodes"].edge_index
        data_edge_index = batch_data["nodes", "data", "nodes"].edge_index
        call_edge_index = batch_data["nodes", "call", "nodes"].edge_index

        # Get number of graphs and graph nodes list
        control_pos_lists = batch_data["nodes", "control", "nodes"].edge_attr
        data_pos_lists = batch_data["nodes", "data", "nodes"].edge_attr
        call_pos_lists = batch_data["nodes", "call", "nodes"].edge_attr

        # Get number of graphs and graph nodes list
        num_graphs = batch_data.num_graphs
        graph_nodes_list = batch_data["nodes"]["batch"]

        # Get the auxiliary features vector
        wgsize_log1p = batch_data["wgsize_log1p"]
        transfer_bytes_log1p = batch_data["transfer_bytes_log1p"]
        aux_features = torch.stack((wgsize_log1p, transfer_bytes_log1p), dim=1)

        # Convert the vocabulary ids into dense vectors
        node_states = self.node_embeddings(node_vocab_ids)

        # Clone the initial states (these are used
        #  in the readout head)
        initial_node_states = node_states.clone()

        # Create a single edge_index and pos_lists list
        edge_index = torch.cat(
            [control_edge_index, data_edge_index, call_edge_index], dim=1
        )
        pos_lists = torch.cat(
            [control_pos_lists, data_pos_lists, call_pos_lists], dim=0
        )

        for i in range(self._message_passing_timesteps):
            if i % 2 == 0:
                node_states = self.mp1(x=node_states, edge_index=edge_index)

            else:
                node_states = self.mp2(x=node_states, edge_index=edge_index)

        logits = self.readout(
            initial_node_states=initial_node_states,
            final_node_states=node_states,
            aux_variables=aux_features,
            num_graphs=num_graphs,
            graph_nodes_list=graph_nodes_list,
        )

        return logits
