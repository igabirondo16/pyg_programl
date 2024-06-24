import torch
from torch import nn


class LinearLayer(nn.Module):
    """Custom implementation of a Linear Layer. It applies
    Dropout to the input for regularization.

    It applies the following equation y = xA^T + b
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        use_bias: bool = True,
        dropout_rate: float = 0.0,
    ):
        """Constructor of the linear layer.

        Args:
            input_size (int): Size of the input vector for the Linear Layer.
            output_size (int): Size of the output vector for the Linear Layer.
            use_bias (bool, optional): Whether to use a learnable bias. Defaults to True.
            dropout_rate (float, optional): Probability of using Dropout. Defaults to 0.0.
                Notice that if dropout_rate == 0.0 it has no regularization effect.
        """
        super().__init__()

        self.__input_size = input_size
        self.__output_size = output_size
        self.__use_bias = use_bias
        self.__dropout_rate = dropout_rate

        self.linear_layer = nn.Linear(
            self.__input_size, self.__output_size, bias=self.__use_bias
        )
        self.dropout_layer = nn.Dropout(self.__dropout_rate)

    def reset_parameters(self):
        """Initializes the layer's parameters."""
        self.linear_layer.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method of the layer. It applies
        a linear transformation to the input vector.

        Args:
            x (torch.Tensor): Input vector.

        Returns:
            torch.Tensor: Linear transformation of the
                input vector.
        """
        x = self.dropout_layer(x)
        x = self.linear_layer(x)
        return x
