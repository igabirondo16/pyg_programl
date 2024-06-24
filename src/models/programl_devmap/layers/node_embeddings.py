import torch
from torch import nn


class NodeEmbeddings(nn.Module):
    """Auxiliary class for the Embedding Layer.
    Note that this implementation does not accept positional Embeddings,
    as these are not used for the heterogeneous device mapping problem.

    It creates a look-up table of dimensions [vocab_size x embedding_size]
    """

    def __init__(self, vocab_size: int, embedding_size: int = 32):
        """Constructor of the embedding layer.

        Args:
            vocab_size (int): Number of different tokens in the vocabulary.
            embedding_size (int, optional): Dimension of the output vectors
                of the lookup-table. Defaults to 32.
        """
        super().__init__()
        self.__vocab_size = vocab_size
        self.__embedding_size = embedding_size
        self.embedding_layer = nn.Embedding(self.__vocab_size, self.__embedding_size)

    def reset_parameters(self):
        """Initializes the embedding layer's parameters."""
        self.embedding_layer.reset_parameters()

    def forward(self, vocab_ids: torch.Tensor) -> torch.Tensor:
        """Forward method of the embedding layer. It transforms
        a tensor of vocabulary token ids into vectors of size
        embedding_size.

        Args:
            vocab_ids (torch.Tensor): Tensor of dimension [num_nodes]
                which describes the token id of each node.

        Returns:
            torch.Tensor: Tensor of dimension [num_nodes x embedding_size]
                that describes the transformation from token id to dense
                vector.
        """
        node_embeddigs = self.embedding_layer(vocab_ids)
        return node_embeddigs
