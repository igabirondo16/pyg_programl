import torch
from torch import nn


class NodeEmbeddings(nn.Module):
    """Auxiliary class for the Embedding Layer.
    Note that this implementation DOES ACCEPT selector Embeddings.

    It creates a look-up table of dimensions [vocab_size x (embedding_size + 2)]
    """

    def __init__(
        self,
        vocab_size: int,
        device: str,
        embedding_size: int = 32,
        selector_embedding_value: float = 50,
    ):
        """Constructor of the embedding layer.

        Args:
            vocab_size (int): Number of different tokens in the vocabulary.
            device (str): Device in which the embeddings are initialized.
            embedding_size (int, optional): Dimension of the output vectors
                of the lookup-table. Defaults to 32.
            selector_embedding_value (int, optional): Value for initializing
                the selector embeddings.
        """
        super().__init__()
        self.__vocab_size = vocab_size
        self.__embedding_size = embedding_size
        self.embedding_layer = nn.Embedding(self.__vocab_size, self.__embedding_size)

        # Selector embeddings
        selector_init = torch.tensor(
            [[selector_embedding_value, 0], [0, selector_embedding_value]],
            dtype=torch.get_default_dtype(),
            device=device,
        )

        # Initialize selector embeddings
        self.selector_embeddings_layer = nn.Embedding.from_pretrained(
            selector_init, freeze=True
        )

    def reset_parameters(self):
        """Initializes the embedding layer's parameters."""
        self.embedding_layer.reset_parameters()

    def forward(
        self, vocab_ids: torch.Tensor, selector_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward method of the embedding layer. It transforms
        tensors of vocabulary token ids and selector ids into vectors of size
        (embedding_size + 2).

        Args:
            vocab_ids (torch.Tensor): Tensor of dimension [num_nodes]
                which describes the token id of each node.
            selector_ids (torch.Tensor): Tensor of dimensions [num_nodes]
                which describes the node that has been taken as root.

        Returns:
            torch.Tensor: Tensor of dimension [num_nodes x (embedding_size + 2)]
                that describes the transformation from token id to dense
                vector.
        """

        node_embeddigs = self.embedding_layer(vocab_ids)
        selector_embeddings = self.selector_embeddings_layer(selector_ids)
        node_embeddigs = torch.cat((node_embeddigs, selector_embeddings), dim=1)

        return node_embeddigs
