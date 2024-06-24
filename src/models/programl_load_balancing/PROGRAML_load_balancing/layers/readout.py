import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool

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
        message_passing_timesteps: int,
        discretize_problem: bool,
        pooling_method: str,
        extended_readout: bool,
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
            message_passing_timesteps (int): Number of message passing iterations
                to perform.
            discretize_problem (bool): Whether to use the model for regression or
                classification. If discretize_problem, the model works for classification.
            pooling_method (str): The pooling method that uses the model. Depending on it
                the input dimensionality changes.
            extended_readout (bool): Whether to use an additional linear layer in the
                feed_forward.
        """
        super().__init__()
        self.num_classes = num_classes
        self._discretize_problem = discretize_problem
        self.pooling_method = pooling_method
        self.message_passing_timesteps = message_passing_timesteps

        # Regression gate takes as input the initial and final node states
        # that is why it has doubled the input size
        # This layers are used by the original pooling method
        self.regression_gate = LinearLayer(
            2 * hidden_size, self.num_classes, use_bias=True, dropout_rate=0.2
        )
        self.regression_transform = LinearLayer(
            hidden_size, self.num_classes, use_bias=True, dropout_rate=0.2
        )

        self.linear_layer = LinearLayer(
            2 * hidden_size, hidden_size, use_bias=True, dropout_rate=0.2
        )

        # Decice the input dimensionality, depending on the pooling method it changes
        if pooling_method == "default":
            initial_size = self.num_classes + aux_features_dim

        elif pooling_method == "skips":
            initial_size = (
                self.message_passing_timesteps * hidden_size + aux_features_dim
            )

        elif pooling_method == "diff" or pooling_method == "topk":
            initial_size = hidden_size + aux_features_dim

        else:
            initial_size = 2 * hidden_size + aux_features_dim

        # If the model is for classification, the output dimensionality is the number of classes.
        if self._discretize_problem:
            output_size = self.num_classes

        else:
            output_size = 1

        self.batch_normalization = nn.BatchNorm1d(initial_size)

        if extended_readout:
            self.feed_forward = nn.Sequential(
                nn.Linear(initial_size, graph_x_layer_size),
                nn.ReLU(),
                nn.Dropout(output_dropout),
                nn.Linear(graph_x_layer_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(output_dropout),
                nn.Linear(hidden_size, output_size),
            )

        else:
            self.feed_forward = nn.Sequential(
                nn.Linear(initial_size, graph_x_layer_size),
                nn.ReLU(),
                nn.Dropout(output_dropout),
                nn.Linear(graph_x_layer_size, output_size),
            )

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

        if self.pooling_method == "default":
            # Create a per-node prediction.
            # Nodewise_readout is a tensor of shape [num_nodes x num_classes]
            gate_input = torch.cat((initial_node_states, final_node_states), dim=-1)
            gate_output = torch.sigmoid(self.regression_gate(gate_input))
            nodewise_readout = gate_output * self.regression_transform(
                final_node_states
            )

            # Aggregate-over all the nodes predictions to their respectives graphs
            # Graph_readout is a tensor of shape [num_graphs x num_classes]
            device = final_node_states.device
            graph_readout = torch.zeros(num_graphs, self.num_classes, device=device)
            graph_readout.index_add_(
                dim=0, index=graph_nodes_list, source=nodewise_readout
            )

        elif self.pooling_method == "mean":
            initial_graph_readout = global_mean_pool(
                initial_node_states, graph_nodes_list
            )
            final_graph_readout = global_mean_pool(final_node_states, graph_nodes_list)

            graph_readout = torch.cat(
                (initial_graph_readout, final_graph_readout), dim=1
            )

        else:
            graph_readout = final_node_states

        if final_node_states.size()[0] > 1:
            # Concatenate the auxiliar variables
            extended_graph_readout = torch.cat((graph_readout, aux_variables), dim=1)

        else:
            extended_graph_readout = torch.cat(
                (graph_readout.view(1, -1), aux_variables.view(1, 2)), dim=1
            )

        # Auxiliar features are in another scale, normalize the tensor
        norm_graph_readout = self.batch_normalization(extended_graph_readout)

        # Make the final prediction per graphs
        logits = self.feed_forward(norm_graph_readout)

        # If regression, flatten the tensor and convert to percentage
        if not self._discretize_problem:
            logits = torch.flatten(logits)

        return logits
