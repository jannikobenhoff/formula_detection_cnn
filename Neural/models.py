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

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=82):
        super(NeuralNetwork, self).__init__()

        # Convolutional layer with 32 filters, kernel size of 3x3, and stride of 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        # Batch normalization layer
        self.bn1 = nn.BatchNorm2d(32)
        # Max pooling layer with kernel size of 2x2 and stride of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Convolutional layer with 64 filters, kernel size of 3x3, and stride of 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        # Batch normalization layer
        self.bn2 = nn.BatchNorm2d(64)
        # Max pooling layer with kernel size of 2x2 and stride of 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Flatten the output from the convolutional layers
        self.fc1 = nn.Linear(1600, 128)
        # Output layer with `num_classes` neurons
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Pass input through the first convolutional layer and activation function
        x = F.relu(self.conv1(x))
        # Pass the output through batch normalization
        x = self.bn1(x)
        # Pass the output through max pooling
        x = self.pool(x)
        # Pass input through the second convolutional layer and activation function
        x = F.relu(self.conv2(x))
        # Pass the output through batch normalization
        x = self.bn2(x)
        # Pass the output through max pooling
        x = self.pool2(x)
        # Flatten the output from the convolutional layers
        x = x.view(x.size(0), -1)
        # Pass the output through the first
        x = F.relu(self.fc1(x))
        # Pass the output through the second fully connected layer to produce the final output
        x = self.fc2(x)
        return x

class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=10 * 6 * 6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=82)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        out = self.fc3(out)

        return out


class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,  # how big is the square that's going over the image?
                      stride=1,  # default
                      padding=1),
            # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)  # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units * 7 * 7,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion
