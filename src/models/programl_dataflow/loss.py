import torch


class WeightedCrossEntropyLoss(torch.nn.Module):
    """Class of a Weighted Cross Entropy Loss"""

    def __init__(self, loss_weighting: float, device: str):
        """Constructor for the Weighted Cross Entropy Loss

        Args:
            loss_weighting (float): Weight for the class 1.
            device (str): Device to initialize the weights,
        """
        super().__init__()
        self.loss_weighting = loss_weighting

        weight = torch.tensor(
            [1.0 - self.loss_weighting, self.loss_weighting], device=device
        )

        self.loss = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-1)

    def forward(self, predictions: torch.tensor, labels: torch.tensor) -> torch.Tensor:
        """Forward method for computing the loss of
        the made predictions.

        Args:
            predictions (torch.tensor): Batch predictions
                made by the model.
            labels (torch.tensor): Ground truth labels
                of the batch.

        Returns:
            torch.Tensor: Loss in the prediction.
        """
        predic_loss = self.loss(predictions, labels)
        return predic_loss
