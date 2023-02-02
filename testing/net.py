from scanning import scale
import torch
import numpy as np
import pandas as pd
import torchvision
import cv2
import scipy
import matplotlib as plt
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torch import nn
import shutil
from tqdm.auto import tqdm
from timeit import default_timer as timer

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
a = 1

class NN(nn.Module):

    def __init__(self):
        super(NN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

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


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
      dir_path (str or pathlib.Path): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

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

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

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

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
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
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)

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


model_Fashion = NN().to(device)

if __name__ == '__main__':
    """get data"""

    train_path = "__files/train_images"
    test_path = "__files/test_images"
    images = os.listdir('__files/train_images/!')
    list_of_files = {}
    all_subdirs = [d for d in os.listdir('__files/train_images') if os.path.isdir(d)]
    subdirs = [x[0] for x in os.walk('__files/train_images')]
    subdirs.pop(0)
    for i in range(len(subdirs)):
        if '\\' in subdirs[i]:
            subdirs[i] = subdirs[i].split('\\')[1]
        else:
            subdirs[i] = subdirs[i].split('\\')[1]
    labels = subdirs
    len_labels = len(labels)

    """convert images"""

    #im = Image.open('__files/extracted_images/'+subdirs[0]+"/"+images[0])
    #print(np.asarray(im))

    #im = scale(np.asarray(im))
    #print(im.shape)

    """Prepare Data"""
    c = 0
    d = 0
    prepare_data = False
    if prepare_data:
        for i in range(len_labels):
            print(labels[i])
            new_sub_dir = labels[i]
            parent = '__files/images_test'
            path = os.path.join(parent, new_sub_dir)
            if not os.path.exists(path):
                os.mkdir(path)
                print("Directory '% s' created" % new_sub_dir)
            dir = os.listdir('__files/train_images/'+labels[i])
            for j in dir:
                c += 1
                #im = scale(np.asarray(Image.open('__files/extracted_images/'+labels[i]+"/"+j)))
                #im = torch.from_numpy(im)
                if not c % 5:
                    source = os.path.join("__files/train_images", labels[i]+"/"+j)
                    destination = '__files/test_images' + labels[i]+"/"+j
                    print(destination)
                    if not os.path.exists(destination):
                        destination = '__files/test_images/'+labels[i]
                        d += 1
                        #dest = shutil.move(source, destination)
                if not c % 10000:
                    print(c)

    walk_through_dir('__files')
    print("Train images: ", c)
    print("Test images: ", d)

    """transform data"""

    augment = False #True for data augmentation
    augment_bins = 31
    pic_size = 28
    if augment:
        data_transformer = transforms.Compose([transforms.Resize(size=(pic_size, pic_size)),
                                               transforms.TrivialAugmentWide(num_magnitude_bins=augment_bins),
                                               transforms.ToTensor()])
    else:
        data_transformer = transforms.Compose([transforms.Resize(size=(pic_size, pic_size)),
                                               transforms.ToTensor()])
    """create dataset"""

    train_data = datasets.ImageFolder(root=train_path, transform=data_transformer, target_transform=None)
    test_data = datasets.ImageFolder(root=test_path, transform=data_transformer, target_transform=None)

    class_names = train_data.classes
    class_dict = train_data.class_to_idx
    print(len(train_data), len(test_data))
    print(train_data)
    img, label = train_data[0][0], train_data[0][1]

    #print(f"Image tensor:\n{img}")
    print(f"Image shape: {img.shape}")
    print(f"Image datatype: {img.dtype}")
    print(f"Image label: {label}")
    print(f"Label datatype: {type(label)}")

    batch_size = 32
    num_workers = os.cpu_count()

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    test_dataloader = DataLoader(dataset=test_data, batch_size=1, num_workers=1, shuffle=False)

    """test models"""

    torch.manual_seed(42)
    model_tiny = TinyVGG(input_shape=3,  # number of color channels (3 for RGB, 1 for BaW)
                         hidden_units=10,
                         output_shape=len(train_data.classes)).to(device)

    models = [model_Fashion, model_tiny]

    select_model = 1

    img, label = next(iter(train_dataloader))
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label.shape}")


    # 1. Get a batch of images and labels from the DataLoader
    img_batch, label_batch = next(iter(train_dataloader))
    # 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
    img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
    print(f"Single image shape: {img_single.shape}\n")

    # 3. Perform a forward pass on a single image
    models[select_model].eval()
    with torch.inference_mode():
        pred = models[select_model](img_single.to(device))

    # 4. Print out what's happening and convert model logits -> pred probs -> pred label
    sym_pred = labels[torch.argmax(torch.softmax(pred, dim=1), dim=1)]
    sym_label = labels[label_single]
    print(f"Output logits:\n{pred}\n")
    print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    print(f"Output prediction label:\n{sym_pred}\n")
    print(f"Actual label:\n{sym_label}")

    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Set number of epochs
    NUM_EPOCHS = 1

    # Recreate an instance of TinyVGG
    model_0 = TinyVGG(input_shape=3,  # number of color channels (3 for RGB)
                      hidden_units=10,
                      output_shape=len(train_data.classes)).to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

    # Start the timer

    start_time = timer()

    # Train model_0
    model_0_results = train(model=model_0,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=NUM_EPOCHS)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")


