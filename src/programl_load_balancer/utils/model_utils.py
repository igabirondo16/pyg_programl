import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

BATCH_SIZE = 256
NUM_EPOCHS = 300
LEARNING_RATE = 1e-3


def save_model_checkpoint(model: torch.nn.Module, discretize_problem: bool):
    """Function to store a checkpoint of the model.

    Args:
        model (torch.nn.Module): Model to save.
        discretize_problem (bool): Whether it is the
            classification or regression model.
    """

    if discretize_problem:
        filename = "lb_checkpoint_disc.pt"

    else:
        filename = "lb_checkpoint_regr.pt"

    checkpoint_filename = os.path.join("./", filename)

    dict_state = {
        "model": model.state_dict(),
    }

    torch.save(dict_state, checkpoint_filename)

    print(f"Savng model checkpoint at: {checkpoint_filename}")


def get_checkpoint(discretize_problem: bool) -> Dict:
    """Function to load a model's checkpoint.

    Args:
        discretize_problem (bool): Whether it is the
            classification or regression model.

    Returns:
        Dict: Checkpoint of the model.
    """
    if discretize_problem:
        path = "lb_checkpoint_disc.pt"

    else:
        path = "lb_checkpoint_regr.pt"

    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path)
    print(f"Model checkpoint succesfully loaded at: {path}")
    return checkpoint


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn,
    optimizer: torch.optim,
    device: str,
) -> float:
    """Function for training one epoch the input model.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (DataLoader): DataLoader with the training
            data.
        criterion (torch.nn): Loss function of the model.
        optimizer (torch.optim): Optimizer of the model's
            parameters.
        device (str): Device to execute the training.

    Returns:
        float: Loss of the training epoch.
    """
    model.train()
    total_loss = 0

    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out_logits = model(data)  # Perform a single forward pass.

        loss = criterion(out_logits, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        # Concat the results of the minibatch
        # to compute metrics afterwards
        total_loss += loss.item()

    return total_loss


def train_production_model(model, dataset, discretize_problem):
    """Function that trains the model used for making the final
    predictions.

    Args:
        model: Load Balancing model to train.
        dataset: Load Balancing dataset.
        discretize_problem: Whether it is the
            classification or regression model.
    """

    print("Starting training process")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Choose the correct loss function
    if discretize_problem:
        criterion = torch.nn.CrossEntropyLoss()

    else:
        criterion = torch.nn.MSELoss()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Reset the model and initialize the optimizer
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Start the training
    for epoch in range(NUM_EPOCHS):
        train_loss = train(
            model=model,
            train_loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        if epoch % 25 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss}")

    # Save the final checkpoint
    print("Training process finished")
    save_model_checkpoint(model, discretize_problem)


def compute_factor(model, heterodata_graph, discretize_problem) -> float:
    """Function that computes the amount of work that the CPU
    should receive for the given OpenCL kernel.

    Args:
        model: Production Load Balancing model.
        heterodata_graph: OpenCL kernel in HeteroData format.
        discretize_problem: Whether it is the
            classification or regression model.

    Returns:
        float: Amount of work that the CPU should receive. This amount
            is given as a ratio.
    """
    model.eval()

    model = model.cpu()
    logits = model(heterodata_graph)

    if discretize_problem:
        probs = F.softmax(logits)

        factor = probs.argmax().item()
        factor *= 0.25

    else:
        factor = logits.clamp(min=0.0, max=100.0).item()
        factor = factor / 100
        factor = round(factor, 3)

    return factor
