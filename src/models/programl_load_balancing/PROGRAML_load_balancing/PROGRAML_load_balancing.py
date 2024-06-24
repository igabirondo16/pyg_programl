import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from .layers.message_passing_layer import MessagePassingLayer
from .layers.node_embeddings import NodeEmbeddings
from .layers.pooling import DenseDiffPooler, TopKPooler
from .layers.readout import Readout


class ProgramlLoadBalancingModel(nn.Module):
    """Replica of the ProGraML model for the load balancing problem.
    This model can use different types of pooling methods, and different
    sizes of readout heads.
    In addition, the implementation is thought for working for both classification
    and regression.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        edge_type_count: int,
        use_edge_bias: bool,
        dropout_rate_linear_layer: float,
        dropout_rate_update_layer: float,
        num_classes: int,
        output_dropout: float,
        aux_features_dim: int,
        graph_x_layer_size: int,
        use_forward_and_backward_edges: bool,
        discretize_problem: bool,
        pooling_method: str,
        extended_readout: bool,
    ):
        """Constructor of the class.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Dimension of the hidden vectors.
            edge_type_count (int): Number of edge types (control, data and call).
            use_edge_bias (bool): Whether to use learnable biases for each edge
                type.
            dropout_rate_linear_layer (float): Probability of using Dropout
                in the linear layer. Notice that if
                dropout_rate_linear_layer == 0.0 it has no regularization effect.
            dropout_rate_update_layer (float): Probability of using Dropout
                in the update layer. Notice that if dropout_rate_update_layer == 0.0
                it has no regularization effect.
            num_classes (int): Number of classes of the dataset.
            output_dropout (float): Probability of using Dropout
                in the readout layer. Notice that if
                output_dropout == 0.0 it has no regularization effect.
            aux_features_dim (int): Dimension of the auxiliar features
                vector.
            graph_x_layer_size (int): Dimension of the graph vector in
                the intermediate layers of the feed_forward layer.
            use_forward_and_backward_edges (bool): Whether to allow messages
                from source to target and from target to source nodes.
            discretize_problem (bool): Whether to use the model for regression or
                classification. If discretize_problem, the model works for classification.
            pooling_method (str): The pooling method that uses the model.
            extended_readout (bool): Whether to use an additional linear layer in the
                feed_forward of the readout.
        """
        super().__init__()

        self.use_forward_and_backward_edges = use_forward_and_backward_edges
        self.pooling_method = pooling_method
        self.message_passing_timesteps = 6
        self.node_embeddings = NodeEmbeddings(vocab_size, hidden_size)

        self.mp1 = MessagePassingLayer(
            hidden_size,
            edge_type_count,
            use_edge_bias,
            use_forward_and_backward_edges,
            dropout_rate_linear_layer,
            dropout_rate_update_layer,
        )

        self.mp2 = MessagePassingLayer(
            hidden_size,
            edge_type_count,
            use_edge_bias,
            use_forward_and_backward_edges,
            dropout_rate_linear_layer,
            dropout_rate_update_layer,
        )

        if self.pooling_method == "topk":
            self.pooling = TopKPooler(hidden_size=hidden_size)

        elif self.pooling_method == "diff":
            self.pooling = DenseDiffPooler(hidden_size=hidden_size)

        else:
            self.pooling = None

        self.readout = Readout(
            num_classes,
            hidden_size,
            output_dropout,
            aux_features_dim,
            graph_x_layer_size,
            self.message_passing_timesteps,
            discretize_problem,
            pooling_method,
            extended_readout,
        )

    def reset_parameters(self):
        """Initializes the layer's parameters."""
        self.node_embeddings.reset_parameters()
        self.mp1.reset_parameters()
        self.mp2.reset_parameters()
        self.readout.reset_parameters()

        if self.pooling is not None:
            self.pooling.reset_parameters()

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
        dense vectors, it makes 6 iterations of message passing
        layers, and using the initial and final node dense vectors
        it makes a prediction per graph in the batch.

        Args:
            batch_data (Batch): Object that saves all the information
                of the graphs in the batch:

                - batch_data['nodes']['x']: Stores the vocab ids for
                    all the nodes in the batchs. Tensor of dimension
                    [num_nodes]

                - batch_data['nodes', '_edge_type', 'nodes'].edge_index:
                    Adjacency matrix of the batch graphs per edge type.

                - batch_data.num_graphs: Number of graphs in the batch.

                - batch_data['nodes']['batch']: Tensor of nodes that indicates
                    to which graph each node of the batch belongs to.
                    Tensor of dimension [num_nodes].

                - batch_data['y']: Tensor containing the label of each graph.
                    Dimension [num_graphs].

                - batch_data['wgsize_log1p']: Auxiliar dynamic feature per graph.
                    Dimension [num_graphs].

                - batch_data['transfer_bytes_log1p']: Auxiliar dynamic feature per graph.
                    Dimension [num_graphs].

        Returns:
            torch.Tensor: Tensor representing the probabilities of each class per
                graph in the batch.
                Dimensions [num_graphs x num_classes].
        """

        # Get the node vocabulary ids
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
        if isinstance(batch_data, Batch):
            num_graphs = batch_data.num_graphs

        else:
            num_graphs = 0

        graph_nodes_list = batch_data["nodes"].get("batch", None)
        if graph_nodes_list is None:
            graph_nodes_list = torch.zeros(node_vocab_ids.size()[0], dtype=torch.int64)

        # Get the auxiliary features vector
        wgsize_log1p = batch_data["wgsize_log1p"]
        transfer_bytes_log1p = batch_data["transfer_bytes_log1p"]

        if torch.is_tensor(wgsize_log1p) and torch.is_tensor(transfer_bytes_log1p):
            aux_features = torch.stack((wgsize_log1p, transfer_bytes_log1p), dim=1)

        else:
            aux_features = torch.tensor(
                [wgsize_log1p, transfer_bytes_log1p], device=node_vocab_ids.device
            ).to(torch.float32)

        # Convert the vocabulary ids into dense vectors
        node_states = self.node_embeddings(node_vocab_ids)

        # Clone the initial states (these are used
        #  in the readout head)
        initial_node_states = node_states.clone()

        # Create a single edge_index list
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

        control_edge_type = torch.empty(
            num_control_edges, device=edge_index.device
        ).fill_(0)
        data_edge_type = torch.empty(num_data_edges, device=edge_index.device).fill_(1)
        call_edge_type = torch.empty(num_call_edges, device=edge_index.device).fill_(2)

        edge_types = torch.cat(
            [control_edge_type, data_edge_type, call_edge_type]
        ).int()

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

        if self.pooling_method == "skips":
            final_node_states = []

        for i in range(self.message_passing_timesteps):
            if i % 2 == 0:
                node_states = self.mp1(node_states, edge_index, pos_lists, edge_sources)

            else:
                node_states = self.mp2(node_states, edge_index, pos_lists, edge_sources)

            if self.pooling_method == "skips":
                mean_graph_embedding = global_mean_pool(node_states, graph_nodes_list)
                final_node_states.append(mean_graph_embedding)

        if self.pooling_method == "skips":
            final_node_states = torch.cat(final_node_states, dim=1)

        elif self.pooling_method == "mean" or self.pooling_method == "default":
            final_node_states = node_states

        else:
            final_node_states = self.pooling(node_states, edge_index, graph_nodes_list)

        # Make the final prediction
        logits = self.readout(
            initial_node_states,
            final_node_states,
            aux_features,
            num_graphs,
            graph_nodes_list,
        )

        return logits
