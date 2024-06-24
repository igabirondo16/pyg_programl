import torch
from torch import nn


class UpdateLayer(nn.Module):
    """Layer for updating node states after aggregating all the
    messages in their respective nodes.

    """

    def __init__(self, input_size: int, dropout_rate: float = 0.0):
        """Constructor of the update layer.

        Args:
            input_size (int): Dimension of the input tensor.
            dropout_rate (float, optional): Probability of using Dropout.
                Defaults to 0.0. Notice that if dropout_rate == 0.0 it has
                no regularization effect.

        """
        super().__init__()
        self.__input_size = input_size
        self.__dropout_rate = dropout_rate

        self.gru_layer = nn.GRUCell(
            input_size=self.__input_size, hidden_size=self.__input_size
        )
        self.dropout_layer = nn.Dropout(self.__dropout_rate)

    def reset_parameters(self):
        """Initializes the layer's parameters."""
        self.gru_layer.reset_parameters()

    def forward(
        self, aggr_out: torch.Tensor, node_states: torch.Tensor
    ) -> torch.Tensor:
        """Forward method for the update layer. It takes as input
        the aggregated messages (one per node) and the node states
        of the last timestamp for creating the node states at time t.

        Args:
            aggr_out (torch.Tensor): Result of the aggregation of
                the messages. Dimension [num_nodes x input_size]
            node_states (torch.Tensor): Node states of timestamp
                t-1. Dimension [num_nodes x input_size]

        Returns:
            torch.Tensor: Updated node states at timestamp t.
                Dimension [num_nodes x input_size]
        """
        output = self.gru_layer(aggr_out, node_states)
        output = self.dropout_layer(output)
        return output
