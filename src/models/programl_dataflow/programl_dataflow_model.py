import torch
import torch_geometric
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.typing import SparseTensor

from .layers.message_passing_layer import MessagePassingLayer
from .layers.node_embeddings import NodeEmbeddings
from .layers.readout import Readout


class ProgramlDataflowModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        selector_embedding_size: int,
        selector_embedding_value: float,
        edge_type_count: int,
        use_edge_bias: bool,
        use_forward_and_backward_edges: bool,
        dropout_rate_linear_layer: float,
        dropout_rate_update_layer: float,
        num_message_passing_layers: int,
        message_passing_timesteps: int,
        num_classes: int,
        output_dropout: float,
        device: str,
        edge_position_max: int = 4096,
    ):
        """Constructor of the ProGraML model.

        Args:
            vocab_size (int): Number of different tokens in the vocabulary.
            hidden_size (int): Dimension of the output vectors
                of the lookup-table.
            selector_embedding_size (int): Dimension of the
                selector embeddings.
            selector_embedding_value (float): Value for initializing
                the selector embeddings.
            edge_type_count (int): Number of edge types (control, data and call).
            use_edge_bias (bool): Whether to use learnable biases for each edge
                type.
            use_forward_and_backward_edges (bool): Whether to allow messages
                from source to target and from target to source nodes.
            dropout_rate_linear_layer (float): Probability of using Dropout
                in the linear layer. Notice that if dropout_rate == 0.0 it has no
                regularization effect.
            dropout_rate_update_layer (float): Probability of using Dropout
                in the update layer. Notice that if dropout_rate == 0.0 it has no
                regularization effect.
            num_message_passing_layers (int): Number of message passing layers of the model.
            message_passing_timesteps (int): Number of timesteps each message passing layer will perform.
            num_classes (int): Number of classes of the training dataset.
            output_dropout (float): Probability of using Dropout
                in the readout layer. Notice that if output_dropout == 0.0
            device (str): Device to store the model.
            edge_position_max (int, optional): Number of positional embeddings
                to create. Defaults to 4096.
        """
        super().__init__()

        # Initialize the node embeddings
        self.node_embeddings = NodeEmbeddings(
            vocab_size, device, hidden_size, selector_embedding_value
        )

        self.num_message_passing_layers = num_message_passing_layers
        self.message_passing_timesteps = message_passing_timesteps
        self.use_forward_and_backward_edges = use_forward_and_backward_edges

        # More than one message passing layer can be stored
        # Create a ModuleList to store them all
        self.message_passing_layers = nn.ModuleList()
        for i in range(num_message_passing_layers):
            self.message_passing_layers.append(
                MessagePassingLayer(
                    hidden_size=hidden_size,
                    selector_embedding_dimensionality=selector_embedding_size,
                    edge_type_count=edge_type_count,
                    use_edge_bias=use_edge_bias,
                    use_forward_and_backward_edges=use_forward_and_backward_edges,
                    dropout_rate_linear_layer=dropout_rate_linear_layer,
                    dropout_rate_update_layer=dropout_rate_update_layer,
                    edge_position_max=edge_position_max,
                )
            )

        # Readout head for classification
        self.readout = Readout(
            num_classes=num_classes,
            hidden_size=hidden_size,
            selector_embedding_dimensionality=selector_embedding_size,
            output_dropout=output_dropout,
        )

    def reset_parameters(self):
        """Initializes the layer's parameters."""
        self.node_embeddings.reset_parameters()

        for i in range(self.num_message_passing_layers):
            self.message_passing_layers[i].reset_parameters()

        self.readout.reset_parameters()

    def __get_edge_source(
        self,
        edge_index: torch.Tensor,
        num_edge_type: torch.Tensor,
        use_forward_and_backward_edges: bool,
    ) -> torch.Tensor:
        """Function to precompute the message sources of all nodes
        from the graph. To each source index, it multiplies the number
        of edges and adds the type of the edge. Doing this precalculation
        enables sending all the messages of the MessagePassing layer
        with a single call of F.embedding().

        Args:
            edge_index (torch.Tensor): Tensor containing the adjacency matrix
                of the graph in a sparse form. It involves call, data and control
                edges in a single tensor.
            num_edge_type (torch.Tensor): Number of edges per type.
            use_forward_and_backward_edges (bool): Whether to allow messages
                from source to target and from target to source nodes.

        Returns:
            torch.Tensor: Message sources with their equivalent added offset.
        """
        edge_sources = edge_index[0, :]

        # Types of edges
        edge_types = 3
        if use_forward_and_backward_edges:
            edge_types = 6

        # Initialize offset
        offset = torch.tensor([], device=edge_index.device)

        # Edges are ordered in control, data, call (always in the same order)
        # Create a tensor for storing the offset that will be added later
        for idx, num_edges in enumerate(num_edge_type):
            temp = torch.empty(num_edges, device=edge_index.device).fill_(idx)
            offset = torch.cat((offset, temp))

        # Compute the relative edge sources
        offset = offset.to(torch.int32)
        edge_sources = edge_sources.mul(edge_types)
        edge_sources = edge_sources.add(offset)

        return edge_sources

    def forward(self, batch_data: Batch) -> torch.Tensor:
        """Forward method of the ProGraML model. It first decomposes
        the batch data, it converts the node vocabulary ids into
        dense vectors, it makes self.message_passing_iterations iterations
        of message passing layers, and using the initial and final node dense
        vectors it makes a prediction per node in the batch.

        Args:
            batch_data (Batch): Object that saves all the information
                of the graphs in the batch:

                - batch_data['nodes']['x']: Stores the vocab ids for
                    all the nodes in the batchs. Tensor of dimension
                    [num_nodes]

                - batch_data['nodes', '_edge_type', 'nodes'].edge_index:
                    Adjacency matrix of the batch graphs per edge type.

                - batch_data['nodes', '_edge_type', 'nodes'].edge_attr:
                    Position attribute for each edge.

                - batch_data.num_graphs: Number of graphs in the batch.

                - batch_data['nodes']['batch']: Tensor of nodes that indicates
                    to which graph each node of the batch belongs to.
                    Tensor of dimension [num_nodes].

                - batch_data['y']: Tensor containing the label of each graph.
                    Dimension [num_graphs].

        Returns:
            torch.Tensor: Tensor representing the probabilities of each class per
                graph in the batch.
                Dimensions [num_graphs x num_classes].
        """

        # Get the node vocabulary and selector ids
        node_vocab_ids = batch_data["nodes"]["x"]
        node_selector_ids = batch_data["nodes"]["selector_ids"]

        # Extract adjacency matrices per edge type
        control_edge_index = batch_data["nodes", "control", "nodes"].edge_index
        data_edge_index = batch_data["nodes", "data", "nodes"].edge_index
        call_edge_index = batch_data["nodes", "call", "nodes"].edge_index

        # Get number of graphs and graph nodes list
        control_pos_lists = batch_data["nodes", "control", "nodes"].edge_attr
        data_pos_lists = batch_data["nodes", "data", "nodes"].edge_attr
        call_pos_lists = batch_data["nodes", "call", "nodes"].edge_attr

        # Convert the vocabulary ids into dense vectors
        node_states = self.node_embeddings(node_vocab_ids, node_selector_ids)

        # Clone the initial states (these are used
        #  in the readout head)
        initial_node_states = node_states.clone()

        # Create a single edge_index and pos_lists list
        edge_index = torch.cat(
            [control_edge_index, data_edge_index, call_edge_index], dim=1
        )
        pos_lists = torch.cat(
            [control_pos_lists, data_pos_lists, call_pos_lists], dim=0
        )  # CHECK THIS

        # Create an auxiliary tensor for retrieving the
        # adjacency matrices per edge type afterwards
        num_control_edges = control_edge_index.shape[1]
        num_data_edges = data_edge_index.shape[1]
        num_call_edges = call_edge_index.shape[1]

        # Tensor storing the number of edges of each edge type
        num_edge_type = torch.tensor(
            [num_control_edges, num_data_edges, num_call_edges],
            device=edge_index.device,
        )

        # Use bidirectional edges
        if self.use_forward_and_backward_edges:

            # Flip the actual edges for getting the bidirectional adjacency matrix
            control_edge_index_rev = torch.flip(control_edge_index, dims=(0,))
            data_edge_index_rev = torch.flip(data_edge_index, dims=(0,))
            call_edge_index_rev = torch.flip(call_edge_index, dims=(0,))

            # Edge index containing bidirectional edges
            edge_index = torch.cat(
                [
                    control_edge_index,
                    data_edge_index,
                    call_edge_index,
                    control_edge_index_rev,
                    data_edge_index_rev,
                    call_edge_index_rev,
                ],
                dim=1,
            )

            # Position attribute remain the same
            pos_lists = torch.cat(
                [
                    control_pos_lists,
                    data_pos_lists,
                    call_pos_lists,
                    control_pos_lists,
                    data_pos_lists,
                    call_pos_lists,
                ],
                dim=0,
            )

            # Number of edges remain the same
            num_edge_type = torch.tensor(
                [
                    num_control_edges,
                    num_data_edges,
                    num_call_edges,
                    num_control_edges,
                    num_data_edges,
                    num_call_edges,
                ],
                device=edge_index.device,
            )

        # Precompute the sources of the edges for getting the messages faster
        edge_sources = self.__get_edge_source(
            edge_index, num_edge_type, self.use_forward_and_backward_edges
        )

        edge_index = edge_index.to(torch.int64)
        pos_lists = pos_lists.to(torch.int64)
        edge_sources = edge_sources.to(torch.int64)

        # Update the node states by sending messages
        for i in range(self.num_message_passing_layers):
            for t in range(self.message_passing_timesteps):
                node_states = self.message_passing_layers[i](
                    node_states, edge_index, pos_lists, edge_sources
                )

        # Make the predictions per node
        logits = self.readout(initial_node_states, node_states)

        return logits
