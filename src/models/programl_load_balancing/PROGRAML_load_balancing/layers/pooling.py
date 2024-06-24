import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (GCNConv, TopKPooling, dense_diff_pool,
                                global_mean_pool)
from torch_geometric.utils import to_dense_adj, to_dense_batch


class GraphConvolution(nn.Module):
    """Auxiliar class that wrapps two GCNs."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        normalize: bool = False,
    ):
        """Constructor of the class

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Dimension of the hidden size.
            out_channels (int): Dimension of the output.
            normalize (bool, optional): Whether to add self-loops
                and compute symmetric normalization coefficients on-the-fly.
                Defaults to False.
        """
        super(GraphConvolution, self).__init__()

        self.gcn1 = GCNConv(in_channels, hidden_channels, normalize=normalize)
        self.gcn2 = GCNConv(hidden_channels, out_channels, normalize=normalize)

    def reset_parameters(self):
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        node_states1 = F.relu(self.gcn1(node_states, edge_index, mask))
        node_states2 = F.relu(self.gcn2(node_states1, edge_index, mask))
        return node_states2


class DenseDiffPooler(nn.Module):
    """Pooling operator that uses the Dense Differentiable Pooling
    for creating the Graph Representation.

    Using this pooling leads to a high memory use, lowering the
    batch size is recommended.
    """

    def __init__(self, hidden_size: int):
        super(DenseDiffPooler, self).__init__()

        self.output_num_nodes = 512

        self.gc1_pool = GraphConvolution(
            in_channels=hidden_size,
            hidden_channels=hidden_size,
            out_channels=self.output_num_nodes,
        )
        self.gc1_embed = GraphConvolution(
            in_channels=hidden_size,
            hidden_channels=hidden_size,
            out_channels=hidden_size,
        )

    def reset_parameters(self):
        self.gc1_pool.reset_parameters()

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        batch_mask: torch.Tensor,
    ):

        s1 = self.gc1_pool(node_states, edge_index)
        x1 = self.gc1_embed(node_states, edge_index)

        adj_matrix = to_dense_adj(edge_index, batch_mask)
        s1, s1_mask = to_dense_batch(s1, batch_mask)
        x1, x1_mask = to_dense_batch(x1, batch_mask)

        x2, edge_index2, l2, e2 = dense_diff_pool(x1, adj_matrix, s1, s1_mask)
        x2 = x2.mean(dim=1)

        return x2


class TopKPooler(nn.Module):
    """Pooling operator that uses the TopK Pooling operator.
    This operator iterates reducing the number of nodes in the graph,
    until the input graph only has a few nodes that contain the information
    of the full graph.
    """

    def __init__(
        self,
        hidden_size: int,
    ):
        super(TopKPooler, self).__init__()

        self.pooling = TopKPooling(hidden_size, ratio=0.8)
        self.gc = GCNConv(hidden_size, hidden_size)

    def reset_parameters(self):
        self.pooling.reset_parameters()
        self.gc.reset_parameters()

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        batch_mask: torch.Tensor,
    ):

        # == STEP 1 ==
        node_states, edge_index, _, batch_mask, _, _ = self.pooling(
            node_states, edge_index, None, batch_mask
        )
        node_states = F.relu(self.gc(node_states, edge_index))

        # == STEP 2 ==
        node_states, edge_index, _, batch_mask, _, _ = self.pooling(
            node_states, edge_index, None, batch_mask
        )
        node_states = F.relu(self.gc(node_states, edge_index))

        # == STEP 3 ==
        node_states, edge_index, _, batch_mask, _, _ = self.pooling(
            node_states, edge_index, None, batch_mask
        )
        node_states = F.relu(self.gc(node_states, edge_index))
        node_states = global_mean_pool(node_states, batch_mask)

        return node_states
