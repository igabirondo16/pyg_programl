import torch
from torch import nn

from .linear_layer import LinearLayer


class Readout(nn.Module):
    """Readout head of the ProGraML model. It takes as input
    the node hidden states, and makes per-node predictions.

    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        selector_embedding_dimensionality: int,
        output_dropout: float,
    ):
        """Constructor of the readout class.

        Args:
            num_classes (int): Number of classes in
                the dataset.
            hidden_size (int): Dimension of the input vector.
            selector_embedding_dimensionality (int):
                Dimensionality of the selector embeddings.
            output_dropout (float): Probability of using Dropout
                in the readout layer. Notice that if
                output_dropout == 0.0 it has no regularization effect.
        """
        super().__init__()
        self.num_classes = num_classes
        self.dimensionality = hidden_size + selector_embedding_dimensionality

        # Regression gate takes as input the initial and final node states
        # that is why it has doubled the input size
        self.regression_gate = LinearLayer(
            2 * self.dimensionality,
            self.num_classes,
            use_bias=True,
            dropout_rate=output_dropout,
        )
        self.regression_transform = LinearLayer(
            self.dimensionality,
            self.num_classes,
            use_bias=True,
            dropout_rate=output_dropout,
        )

    def reset_parameters(self):
        """Initializes the layer's parameters."""
        self.regression_gate.reset_parameters()
        self.regression_transform.reset_parameters()

    def forward(
        self,
        initial_node_states: torch.Tensor,
        final_node_states: torch.Tensor,
    ):
        """Forward method of the readout layer. It takes as input
        the initial node states and the final node states, and it
        creates a per node prediction.

        Args:
            initial_node_states (torch.Tensor): Node hidden states
                at timestamp t0. Dimension [num_nodes x hidden_size].
            final_node_states (torch.Tensor): Node hidden states after
                passing through all the Message Passing Layers.
                Dimensions [num_nodes x hidden_size].

        Returns:
            torch.Tensor: Tensor representing the probabilities of each
                class per node in the batch.
                Dimensions [num_nodes x num_classes].
        """

        # Create a per-node prediction.
        # Nodewise_readout is a tensor of shape [num_nodes x num_classes]
        gate_input = torch.cat((initial_node_states, final_node_states), dim=-1)
        gate_output = torch.sigmoid(self.regression_gate(gate_input))
        nodewise_readout = gate_output * self.regression_transform(final_node_states)

        return nodewise_readout
