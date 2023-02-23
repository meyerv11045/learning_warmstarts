import json
import logging
import statistics

import torch
import torch.nn as nn
import torch.optim as optim

from learning_warmstarts.dataset.obstacles import generate_obstacles
from learning_warmstarts.losses import IpoptLoss
from learning_warmstarts.utils.io import write_obj

def train_baseline(model, train_loader, val_loader, epochs, learning_rate):
    """
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
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(dev)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()

        for inputs, targets in train_loader:
            inputs = inputs.to(dev)
            targets = targets.to(dev)

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
                inputs = inputs.to(dev)
                targets = targets.to(dev)

                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
            val_losses.append(val_loss / len(val_loader))

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}"
        )

    return model, train_losses, val_losses


def train_ipopt_loss(model, max_iters, convergence_threshold, learning_rate, max_steps, cfg):
    """
    Args:
        model (torch.nn.Module): The model to train.
        max_iters (int):
        convergence_threshold (flaot): 0.001
        learning_rate (float): The learning rate to use asdfor optimization.
        max_steps (int): max update steps (in case algo never converges)

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
    while not converged and i < max_steps:
        inputs = torch.tensor(generate_obstacles(cfg))

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, inputs)
        losses.append(loss.item())
        loss.backward()

        optimizer.step()

        i += 1
        converged = (
            i > 9
            and statistics.stdev(losses) < convergence_threshold * losses[(i - 1) % 10]
        )
        if i % 100 == 0:
            logging.info('%d| average loss: %0.3f', i, statistics.mean(losses[-100:]))

    if converged:
        logging.info('training converged')
    else:
        logging.info('max update steps reached')

    return model, losses

def save_results(model, losses, save_folder, args):
    torch.save(model.state_dict(), save_folder/"model.pt")

    cfg_file = save_folder/"cfg.json"
    cfg_file.write_text(json.dumps(vars(args)))

    write_obj(losses, save_folder/'losses.pkl')
    logging.info('model, losses, and cfg saved.')