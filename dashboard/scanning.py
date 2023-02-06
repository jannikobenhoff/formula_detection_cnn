import numpy as np
import pandas as pd
import scipy.signal
import torchvision
from PIL import Image
import scipy
import matplotlib.pyplot as plt
import cv2


class Zahl():
    def __init__(self, imagearray):
        super(Zahl, self).__init__()
        self.imagearray = imagearray

    def flat(self):
        return self.imagearray.reshape(-1, 28*28)


def baw(imagearray):
    # imagearray[imagearray < 70] = 0
    # imagearray[imagearray > 0] = 255
    """black: 0, white: 255"""
    maxi = imagearray.max()
    mini = imagearray.min()
    print("Max: ", maxi)
    print("Min: ", mini)
    imagearray[imagearray > 60] = 0
    imagearray[imagearray > 0] = 255
    return imagearray


def wab(imagearray):
    imagearray[imagearray < 250] = 0
    return imagearray

def scanning(imagearray):
    zahldrin = []
    zahlen = []
    '''Spalten'''
    start = 0
    end = 0
    for i in range(len(imagearray[0])):
        if sum(imagearray[:, i]) > 0:  # 2040:
            if start == 0:
                start = i
            zahldrin.append(imagearray[:, i])
        elif start != 0:
            end = i
        if end != 0 and start != 0:
            zahlen.append(np.transpose(np.array(zahldrin)))
            zahldrin = []
            start = 0
            end = 0

    zahlen2 = []
    '''Reihen'''
    # for zahl in zahlen:
    #     start = 0
    #     end = 0
    #     for i in range(len(zahl)):
    #         if sum(zahl[i, :]) > 0 and sum(zahl[i+1, :]) > 0:
    #             start = i
    #             break
    #     for i in range(len(zahl)-1, 0, -1):
    #         if sum(zahl[i, :]) > 0 and sum(zahl[i-1, :]) > 0:
    #             end = i
    #             break
    #     zahlen2.append(zahl[start:end, :])
    endlist = []
    startlist = []
    start = 0
    end = 0
    for zahl in zahlen:
        for i in range(len(zahl)):
            if sum(zahl[i, :]) > 0:
                if start == 0:
                    startlist.append(i)
                    start = i
                #zahldrin.append(zahl[i, :])
            elif start != 0:
                end = i
                endlist.append(i)
            if end != 0 and start != 0:
                #zahlen2.append(np.array(zahldrin))
                #zahldrin = []
                start = 0
                end = 0
    heightlist = [x - i for x, i in zip(endlist, startlist)]
    max_index = heightlist.index(max(heightlist))
    for zahl in zahlen:
        # zahlen2.append(zahl[startlist[max_index]:endlist[max_index],:])
        zahlen2.append(zahl[min(startlist):max(endlist), :])

    start = 0
    end = 0
    zahldrin = []
    zahlen3 = []
    for zahl in zahlen2:
        start = 0
        end = 0
        for i in range(len(zahl)):
            if sum(zahl[i, :]) > 0 and sum(zahl[i+1, :]) > 0:
                start = i
                break
        for i in range(len(zahl)-1, 0, -1):
            if sum(zahl[i, :]) > 0 and sum(zahl[i-1, :]) > 0:
                end = i
                break
        zahlen3.append(zahl[start:end, :])

    delete = []
    for i in range(len(zahlen3)):
        zahl = zahlen3[i]
        if min(zahl.shape)/max(zahl.shape) < 0.1:
            delete.append(i)

    for i in reversed(delete):
        del zahlen3[i]
    print(endlist, startlist, heightlist)
    return zahlen3


def scale(imagearray):
    breite = len(imagearray[0])
    höhe = len(imagearray)
    anzahl = int(np.ceil(breite / höhe))
    resized_image = cv2.resize(imagearray, (28*anzahl, 28))
    return resized_image


def addBorder(imagearray, reverse=False):
    sidelength = max(imagearray.shape)
    border = int(sidelength*0.4)
    sidelength = border + sidelength
    print(imagearray.shape, sidelength)

    if sidelength < 25:
        return np.empty(shape=0)
    out = np.zeros([sidelength, sidelength], dtype=np.uint8)
    if reverse:
        out[out == 0] = 255

    x_start, y_start = int((sidelength-imagearray.shape[0])/2), int((sidelength-imagearray.shape[1])/2)
    out[x_start:x_start + imagearray.shape[0], y_start:y_start + imagearray.shape[1]] = imagearray

    # x = np.pad(imagearray, pad_width=10, mode='constant', constant_values=color)
    return out


def scan_process(img_file, plot=True, save=False):
    """
    Input: .JPG Foto
    Output: List array mit Zahlen() Elementen
    """
    image = Image.open(img_file).convert('L')
    if save:
        image.save("screen.jpg")
    image = np.array(image)
    image = baw(image)

    # plt.imshow(image, cmap="gray")
    # plt.show()

    imageList = scanning(image)
    zahlenListe = []
    for image in imageList:
        image = addBorder(image)
        if image != np.empty(shape=0):
            image = scale(image)
            # image = wab(image)
            '''Schwarze Zahl, weißer Hintergrund'''
            image = np.invert(image)
            zahlenListe.append(Zahl(image))

    # if plot:
    #     fig, axes = plt.subplots(1, len(zahlenListe))
    #
    #     for i in range(len(zahlenListe)):
    #         axes[i].imshow(zahlenListe[i].imagearray, cmap="gray")
    #         axes[i].get_yaxis().set_visible(False)
    #         axes[i].get_xaxis().set_visible(False)
    #     plt.savefig("plt.pdf")

    return zahlenListe

if __name__ == "__main__":
    img = Image.open("../testing/__files/zahl4.jpg").convert('L')

    #img = np.array(img)
    # plt.imshow(img, cmap="gray")
    # plt.show()

    zahlen = scan_process("../testing/__files/IMG_1003.jpg")

    fig, axes = plt.subplots(1, len(zahlen))

    for i in range(len(zahlen)):
        axes[i].imshow(zahlen[i].imagearray, cmap="gray")
        axes[i].get_yaxis().set_visible(False)
        axes[i].get_xaxis().set_visible(False)
    plt.show()