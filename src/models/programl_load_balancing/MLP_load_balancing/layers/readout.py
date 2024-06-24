import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

from .linear_layer import LinearLayer


class Readout(nn.Module):
    """Readout head of the ProGraML model. It takes as input
    the node hidden states, and makes a per-graph prediction.
    Note that this readout head is not thought for making
    per-node predictions.

    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        output_dropout: float,
        aux_features_dim: int,
        graph_x_layer_size: int,
        discretize_problem: bool,
    ):
        """Constructor of the readout class.

        Args:
            num_classes (int): Number of classes in
                the dataset.
            hidden_size (int): Dimension of the input vector.
            output_dropout (float): Probability of using Dropout
                in the readout layer. Notice that if
                output_dropout == 0.0 it has no regularization effect.
            aux_features_dim (int): Dimension of the auxiliar features
                vector.
            graph_x_layer_size (int): Dimension of the graph vector in
                the intermediate layers of the feed_forward layer.
            discretize_problem (bool): Whether to use the model for regression or
                classification. If discretize_problem, the model works for classification.
        """
        super().__init__()
        self._num_classes = num_classes
        self._discretize_problem = discretize_problem
        self._hidden_size = hidden_size

        if self._discretize_problem:
            self.feed_forward = nn.Sequential(
                nn.Linear(self._hidden_size + aux_features_dim, graph_x_layer_size),
                nn.ReLU(),
                nn.Dropout(output_dropout),
                nn.Linear(graph_x_layer_size, self._num_classes),
            )
        else:
            self.feed_forward = nn.Sequential(
                nn.Linear(self._hidden_size + aux_features_dim, graph_x_layer_size),
                nn.ReLU(),
                nn.Dropout(output_dropout),
                nn.Linear(
                    graph_x_layer_size, 1
                ),  # Notice that we are making regression, output a single value
            )

        self.batch_normalization = nn.BatchNorm1d(self._hidden_size + aux_features_dim)

    def reset_parameters(self):
        """Initializes the layer's parameters."""
        self.batch_normalization.reset_parameters()

    def forward(
        self,
        node_states: torch.Tensor,
        batch_nodes: torch.Tensor,
        aux_variables: torch.Tensor,
    ):
        """Forward method of the readout layer. It takes as input
        the initial node states, the final node states and the
        auxiliar feature vector, and it creates a per graph
        prediction.

        Args:
            initial_node_states (torch.Tensor): Node hidden states
                at timestamp t0. Dimension [num_nodes x hidden_size].
            final_node_states (torch.Tensor): Node hidden states after
                passing through all the Message Passing Layers.
                Dimensions [num_nodes x hidden_size].
            aux_variables (torch.Tensor): Auxiliar features containing
                dynamic information of the kernels.
                Dimension [aux_features_dim].
            num_graphs (int): Number of graphs in the batch.
            graph_nodes_list (list): List representing to which graph
                each node of the batch belongs to.

        Returns:
            torch.Tensor: Tensor representing the probabilities of each
                class per graph in the batch.
                Dimensions [num_graphs x num_classes].
        """

        # Create a graph readout
        graph_readout = global_mean_pool(node_states, batch_nodes)

        # Concatenate the auxiliar variables
        extended_graph_readout = torch.cat((graph_readout, aux_variables), dim=1)

        # Auxiliar features are in another scale, normalize the tensor
        norm_graph_readout = self.batch_normalization(extended_graph_readout)

        # Make the final prediction per graphs
        logits = self.feed_forward(norm_graph_readout)

        if not self._discretize_problem:
            logits = torch.flatten(logits)

        return logits
