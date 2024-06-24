import torch
from torch import nn

from .layers.node_embeddings import NodeEmbeddings
from .layers.readout import Readout


class MLPLoadBalancing(nn.Module):
    """Load Balancing model that does not use GNN layers."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_classes: int,
        output_dropout: float,
        aux_features_dim: int,
        graph_x_layer_size: int,
        discretize_problem: bool,
    ):
        """Constructor of the MLPLoadBalancing.

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
            discretize_problem (bool): Whether to use the model for regression or
                classification. If discretize_problem, the model works for classification.
        """
        super().__init__()
        self.node_embeddings = NodeEmbeddings(
            vocab_size=vocab_size, embedding_size=hidden_size
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
        self.readout.reset_parameters()

    def forward(self, batch_data):
        # Get the node vocabulary ids
        node_vocab_ids = batch_data["nodes"]["x"]

        # Get the corresponding nodes
        graph_nodes_list = batch_data["nodes"]["batch"]

        # Get the auxiliary features vector
        wgsize_log1p = batch_data["wgsize_log1p"]
        transfer_bytes_log1p = batch_data["transfer_bytes_log1p"]
        aux_features = torch.stack((wgsize_log1p, transfer_bytes_log1p), dim=1)

        # Convert the vocabulary ids into dense vectors
        node_states = self.node_embeddings(node_vocab_ids)

        # Make the final prediction, notice that we don't use GNNs
        logits = self.readout(node_states, graph_nodes_list, aux_features)

        return logits
