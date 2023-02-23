import statistics

import torch
import torch.nn as nn
import torch.optim as optim

from learning_warmstarts.dataset.obstacles import generate_obstacles
from learning_warmstarts.losses import IpoptLoss

# torch.save(model.state_dict(), model_file)


def train_baseline(model, train_loader, val_loader, epochs, learning_rate):
    """
    Trains a regression model using PyTorch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate to use for optimization.

    Returns:
        model (torch.nn.Module): The trained model.
        train_losses (list): List of training losses at the end of each epoch.
        val_losses (list): List of validation losses at the end of each epoch.
    """

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()

        for inputs, targets in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            train_losses.append(loss.item())
            loss.backward()

            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
            val_losses.append(val_loss / len(val_loader))

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}"
        )

    return model, train_losses, val_losses


def train_ipopt_loss(model, max_iters, convergence_threshold, learning_rate, cfg):
    """
    Args:
        model (torch.nn.Module): The model to train.
        max_iters (int):
        convergence_threshold (flaot): 0.001
        learning_rate (float): The learning rate to use asdfor optimization.

    Returns:
        model (torch.nn.Module): The trained model.
        train_losses (list): List of training losses at the end of each epoch.
        val_losses (list): List of validation losses at the end of each epoch.
    """

    criterion = IpoptLoss(max_iters, cfg)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    losses = []
    i = 0
    converged = False
    while not converged:
        inputs = generate_obstacles(cfg)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(inputs, outputs)
        losses.append(loss.item())
        loss.backward()

        optimizer.step()

        i += 1
        converged = (
            i > 9
            and statistics.stdev(losses) < convergence_threshold * loss[(i - 1) % 10]
        )

    return model, losses
