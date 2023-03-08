import torch
import numpy as np
import pandas as pd
import torchvision
import cv2
import scipy
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torch import nn
import shutil
from tqdm.auto import tqdm
from timeit import default_timer as timer
from torchvision.transforms import functional
import torch.nn.functional as F


def train_step(device, model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer, iterations: int = 5):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    count = 0
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        count += 1
        if count == iterations:
            break
    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / iterations
    train_acc = train_acc / iterations

    return train_loss, train_acc


def test_step(device, model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module, iterations: int = 5):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    count = 0
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

            count += 1
            if count == iterations:
                break
    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / iterations
    test_acc = test_acc / iterations

    return test_loss, test_acc


def train_help(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5, test_it: int = 5, train_it: int = 5):
    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer, iterations=train_it)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn, iterations=test_it)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results


def train(model, train_dataloader, test_dataloader, optimizer, epochs=1, adjust_iterations=False, save_model=True,
          train_it=1000, test_it=500, loss_fn=nn.CrossEntropyLoss(), lr=0.001, seed=42):

    train_model = True
    if train_model:
        # Set random seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Set number of epochs
        NUM_EPOCHS = epochs

        # Setup loss function and optimizer
        # loss_fn = nn.CrossEntropyLoss()
        optimizer = optimizer(params=model.parameters(), lr=lr)

        # Start the timer

        start_time = timer()

        # train 1000 test 500 lr = 0.001 -> 194s acc 87
        # train 1000 test 500 lr = 0.01 -> doesn't work
        # train 1000 test 500 lr = 0.001 augmentation_bin = 31 -> 6 176s
        # Adjust number of iterations
        # adjust_iterations = True

        if adjust_iterations:
            train_it = train_it
            test_it = test_it
        else:
            train_it = len(train_dataloader)
            test_it = len(test_dataloader)

        model_0_results = train_help(model=model, train_dataloader=train_dataloader,
                                     test_dataloader=test_dataloader,
                                     optimizer=optimizer,
                                     loss_fn=loss_fn,
                                     epochs=NUM_EPOCHS, train_it=train_it, test_it=test_it)

        # End the timer and print out how long it took
        end_time = timer()
        print(f"Total training time: {end_time - start_time:.3f} seconds")

        # save model
        if save_model:
            m = torch.jit.script(model)
            m.save("model.torch")
        else:
            m = torch.jit.script(model)
            m.save("model_fashion.torch")

    else:
        model = torch.jit.load

        return model
