import torch
from torch import nn

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
        """
        super().__init__()
        self.num_classes = num_classes

        # Regression gate takes as input the initial and final node states
        # that is why it has doubled the input size
        self.regression_gate = LinearLayer(
            2 * hidden_size, self.num_classes, use_bias=True, dropout_rate=0.2
        )
        self.regression_transform = LinearLayer(
            hidden_size, self.num_classes, use_bias=True, dropout_rate=0.2
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(self.num_classes + aux_features_dim, graph_x_layer_size),
            nn.ReLU(),
            nn.Dropout(output_dropout),
            nn.Linear(graph_x_layer_size, self.num_classes),
        )

        self.batch_normalization = nn.BatchNorm1d(self.num_classes + aux_features_dim)

    def reset_parameters(self):
        """Initializes the layer's parameters."""
        self.regression_gate.reset_parameters()
        self.regression_transform.reset_parameters()
        self.batch_normalization.reset_parameters()

    def forward(
        self,
        initial_node_states: torch.Tensor,
        final_node_states: torch.Tensor,
        aux_variables: torch.Tensor,
        num_graphs: int,
        graph_nodes_list: list,
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

        # Create a per-node prediction.
        # Nodewise_readout is a tensor of shape [num_nodes x num_classes]
        gate_input = torch.cat((initial_node_states, final_node_states), dim=-1)
        gate_output = torch.sigmoid(self.regression_gate(gate_input))
        nodewise_readout = gate_output * self.regression_transform(final_node_states)

        # Aggregate-over all the nodes predictions to their respectives graphs
        # Graph_readout is a tensor of shape [num_graphs x num_classes]
        device = final_node_states.device
        graph_readout = torch.zeros(num_graphs, self.num_classes, device=device)
        graph_readout.index_add_(dim=0, index=graph_nodes_list, source=nodewise_readout)

        # Concatenate the auxiliar variables
        extended_graph_readout = torch.cat((graph_readout, aux_variables), dim=1)

        # Auxiliar features are in another scale, normalize the tensor
        norm_graph_readout = self.batch_normalization(extended_graph_readout)

        # Make the final prediction per graphs
        logits = self.feed_forward(norm_graph_readout)

        return logits
