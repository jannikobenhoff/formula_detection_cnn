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


def shutil_or_just_labels(train_path, test_path, shutil=False):

    # train_path = "__files/train_images"
    # test_path = "__files/test_images"

    sub_dirs = [x[0] for x in os.walk('__files/train_images')]
    sub_dirs.pop(0)
    for i in range(len(sub_dirs)):
        if '\\' in sub_dirs[i]:
            sub_dirs[i] = sub_dirs[i].split('\\')[1]
        else:
            sub_dirs[i] = sub_dirs[i].split('\\')[1]
    labels = sub_dirs
    len_labels = len(labels)

    """convert images"""

    #im = Image.open('__files/extracted_images/'+subdirs[0]+"/"+images[0])
    #print(np.asarray(im))

    #im = scale(np.asarray(im))
    #print(im.shape)

    """Prepare Data"""
    c = 0
    d = 0
    if shutil:
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
                        dest = shutil.move(source, destination)
                if not c % 10000:
                    print(c)
        print("Train images: ", c)
        print("Test images: ", d)

    # Show directory system

    #s#how_directories = False

    #if show_directories:
     #   walk_through_dir('__files')

    return labels


def transform_and_load_data(train_path, test_path, batch_size, augment=False, rgb=False, pic_size=(28, 28), augment_bins=31):
    """transform data"""

    # augment = False   # True for data augmentation
    # rgb = False
    # augment_bins = 31
    # pic_size = 28

    if rgb:
        channels = 3
        if augment:
            data_transformer = transforms.Compose([transforms.Resize(size=pic_size),
                                                   transforms.TrivialAugmentWide(num_magnitude_bins=augment_bins),
                                                   transforms.ToTensor()])
        else:
            data_transformer = transforms.Compose([transforms.Resize(size=pic_size),
                                                   transforms.ToTensor()])
    else:
        channels = 1
        if augment:
            data_transformer = transforms.Compose([transforms.Resize(size=(pic_size, pic_size)), transforms.Grayscale(),
                                                   transforms.TrivialAugmentWide(num_magnitude_bins=augment_bins),
                                                   transforms.ToTensor()])
        else:
            data_transformer = transforms.Compose([transforms.Resize(size=(pic_size, pic_size)), transforms.Grayscale(),
                                                   transforms.ToTensor()])
    """create dataset"""

    train_data = datasets.ImageFolder(root=train_path, transform=data_transformer, target_transform=None)
    test_data = datasets.ImageFolder(root=test_path, transform=data_transformer, target_transform=None)

    class_names = train_data.classes
    class_dict = train_data.class_to_idx
    print(len(train_data), len(test_data))
    print(train_data)
    print(class_dict)
    img, label = train_data[0][0], train_data[0][1]

    #print(f"Image tensor:\n{img}")

    print(f"Image shape: {img.shape}")
    print(f"Image datatype: {img.dtype}")
    print(f"Image label: {label}")
    print(f"Label datatype: {type(label)}")

    batch_size = batch_size

    num_workers = os.cpu_count()

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    test_dataloader = DataLoader(dataset=test_data, batch_size=1, num_workers=1, shuffle=False)

    return train_dataloader, test_dataloader
