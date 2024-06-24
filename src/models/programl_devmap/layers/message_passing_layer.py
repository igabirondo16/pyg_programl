import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn
from torch_geometric.nn import MessagePassing

from .linear_layer import LinearLayer
from .position_embeddings import PositionEmbeddings
from .update_layer import UpdateLayer


class MessagePassingLayer(MessagePassing):
    """Implementation of the Message Passing layer described
    in the ProGraML paper. It creates a dense graph representation
    by sending messages between the nodes.

    """

    propagate_type = {
        "transformed_node_states": torch.Tensor,
        "node_states": torch.Tensor,
        "edge_sources": torch.Tensor,
        "pos_lists": torch.Tensor,
        "pos_gating": torch.Tensor,
    }

    edge_type_count: int

    def __init__(
        self,
        hidden_size: int,
        edge_type_count: int,
        use_edge_bias: bool,
        use_forward_and_backward_edges: bool,
        dropout_rate_linear_layer: float = 0.0,
        dropout_rate_update_layer: float = 0.0,
        edge_position_max: int = 4096,
    ):
        """Constructor of the Message Passing layer.

        Args:
            hidden_size (int): Dimension of the input vector.
            edge_type_count (int):  Number of edge types (control, data and call).
            use_edge_bias (bool):Whether to use learnable biases for each edge
                type.
            use_forward_and_backward_edges (bool): Whether to allow messages
                from source to target and from target to source nodes.
            dropout_rate_linear_layer (float, optional): Probability of using Dropout
                in the linear layer. Notice that if dropout_rate == 0.0 it has no
                regularization effect. Defaults to 0.0.
            dropout_rate_update_layer (float, optional): Probability of using Dropout
                in the update layer. Notice that if dropout_rate == 0.0 it has no
                regularization effect. Defaults to 0.0.
            edge_position_max (int, optional): Number of positional embeddings
                to create. Defaults to 4096.
        """

        super().__init__(aggr="mean", flow="source_to_target", node_dim=-2)

        self.__hidden_size = hidden_size
        self.edge_type_count = (
            edge_type_count * 2 if use_forward_and_backward_edges else edge_type_count
        )

        self.linear_net = LinearLayer(
            self.__hidden_size,
            self.__hidden_size * self.edge_type_count,
            use_edge_bias,
            dropout_rate_linear_layer,
        )
        self.gru = UpdateLayer(self.__hidden_size, dropout_rate_update_layer)

        self.pos_transform = LinearLayer(
            self.__hidden_size,
            self.__hidden_size,
            use_edge_bias,
            dropout_rate_linear_layer,
        )
        self.edge_position_max = edge_position_max
        self.register_buffer(
            "position_embs",
            PositionEmbeddings()(
                torch.arange(edge_position_max, dtype=torch.get_default_dtype()),
                demb=self.__hidden_size,
            ),
        )

    def reset_parameters(self):
        """Initializes the layer's parameters."""
        self.linear_net.reset_parameters()
        self.gru.reset_parameters()
        self.pos_transform.reset_parameters()

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        pos_lists: torch.Tensor,
        edge_sources: torch.Tensor,
    ) -> torch.Tensor:
        """Forward method of the layer. It first transforms the current
        node states by their edge type. Then it propagates the messages
        all over the nodes of the graphs. After that it mean-aggregates
        the messages that are forwarded to each node. Finally, it updates
        the node states by using a GRU (gated recurrent unit) layer.

        Args:
            node_states (torch.Tensor): Tensor containing the node hidden
                states at timestamp t - 1. Dimension [num_nodes x hidden_size].
            edge_index (torch.Tensor): Tensor containing the adjacency matrix
                of the graph in a sparse form. It involves call, data and control
                edges in a single tensor.
                Dimension [2 x (num_call_edges + num_data_edges + num_control_edges)]
            pos_lists (torch.Tensor): Position attribute for each of the edges.
                Dimension [1 x (num_call_edges + num_data_edges + num_control_edges)]
            edge_sources (torch.Tensor): Nodes that send the messages (first row
                of edge_index). This tensor has been previously modified so that
                messages can be sent by calling to F.embedding() once.

        Returns:
            torch.Tensor: Tensor containing the node hidden states at timestamp t.
                Dimension [num_nodes x hidden_size].
        """

        # Linearly transform node states. It creates a tensor
        # of dimensions [num_nodes x 3 * hidden_size]
        transformed_node_states = self.linear_net(node_states)

        # Create position embeddings
        pos_gating = 2 * torch.sigmoid(self.pos_transform(self.position_embs))

        # Reshape node_states to [(num_nodes x edge_type) x hidden_size]
        transformed_node_states = transformed_node_states.view(
            node_states.size(0) * self.edge_type_count, self.__hidden_size
        )

        # Start propagating messages. It calls to message(),
        # aggregate() and update().
        final_node_states = self.propagate(
            edge_index=edge_index,
            size=(node_states.size(0), node_states.size(0)),
            transformed_node_states=transformed_node_states,
            node_states=node_states,
            edge_sources=edge_sources,
            pos_lists=pos_lists,
            pos_gating=pos_gating,
        )

        return final_node_states

    def message(
        self,
        edge_index: torch.Tensor,
        transformed_node_states: torch.Tensor,
        node_states: torch.Tensor,
        edge_sources: torch.Tensor,
        pos_lists: torch.Tensor,
        pos_gating: torch.Tensor,
    ) -> torch.Tensor:
        """Function for creating the messages that will be sent to
        the nodes. It creates a tensor of shape [num_edges x hidden_size]
        that will be used as input for the aggregation function.

        Args:
            edge_index (torch.Tensor): Tensor containing the adjacency matrix
                of the graph in a sparse form. It involves call, data and control
                edges in a single tensor.
                Dimension [2 x (num_call_edges + num_data_edges + num_control_edges)]
            transformed_node_states (torch.Tensor): Tensor that contains the
                previously linearly transformed node hidden states.
                Dimensions [3 x hidden_size x num_nodes].
            node_states (torch.Tensor): Tensor containing the node hidden
                states at timestamp t - 1. Dimension [num_nodes x hidden_size].
            edge_sources (torch.Tensor):  Nodes that send the messages (first row
                of edge_index). This tensor has been previously modified so that
                messages can be sent by calling to F.embedding() once.
            pos_lists (torch.Tensor): Position attribute for each of the edges.
                Dimension [1 x (num_call_edges + num_data_edges + num_control_edges)]
            pos_gating (torch.Tensor): Position embeddings.

        Returns:
            torch.Tensor: Tensor containing a message per edge (for all the edge
                types) in the graph.
                Dimension [num_edges x hidden_size]
        """
        # Perform a single call to F.embedding()
        messages = F.embedding(edge_sources, transformed_node_states)

        # Check if the positions are less than self.edge_position_max
        # if not (pos_lists > self.edge_position_max).any():
        #    pos_factor = F.embedding(pos_lists, pos_gating)
        #    messages = messages.mul(pos_factor)

        return messages

    def update(
        self,
        aggr_out: torch.Tensor,
        transformed_node_states: torch.Tensor,
        node_states: torch.Tensor,
        edge_sources: torch.Tensor,
        pos_lists: torch.Tensor,
        pos_gating: torch.Tensor,
    ) -> torch.Tensor:
        """Update function for the node states.
        It takes as input the result of mean-aggregating
        the edge messages, and it creates the next node
        hidden states using a GRU (gated recurrent unit)
        cell.

        Args:
            edge_index (torch.Tensor): Tensor containing the adjacency matrix
                of the graph in a sparse form. It involves call, data and control
                edges in a single tensor.
                Dimension [2 x (num_call_edges + num_data_edges + num_control_edges)]
            transformed_node_states (torch.Tensor): Tensor that contains the
                previously linearly transformed node hidden states.
                Dimensions [3 x hidden_size x num_nodes].
            node_states (torch.Tensor): Tensor containing the node hidden
                states at timestamp t - 1. Dimension [num_nodes x hidden_size].
            edge_sources (torch.Tensor):  Nodes that send the messages (first row
                of edge_index). This tensor has been previously modified so that
                messages can be sent by calling to F.embedding() once.
            pos_lists (torch.Tensor): Position attribute for each of the edges.
                Dimension [1 x (num_call_edges + num_data_edges + num_control_edges)]
            pos_gating (torch.Tensor): Position embeddings.

        Returns:
            torch.Tensor: Tensor containing the node hidden states at timestamp
            t. Dimension [num_nodes x hidden_size].
        """
        update_output = self.gru(aggr_out, node_states)
        return update_output
