import numpy as np
import pandas as pd
import scipy.signal
import torchvision
from PIL import Image
import scipy
import matplotlib.pyplot as plt
import cv2
import torch
from sklearn.svm import SVC
import requests


class Zahl():
    def __init__(self, imagearray):
        super(Zahl, self).__init__()
        self.imagearray = imagearray

    def flat(self):
        return self.imagearray.reshape(-1, 28*28)


def scale(imagearray):
    breite = len(imagearray[0])
    höhe = len(imagearray)
    anzahl = int(np.ceil(breite / höhe))
    resized_image = cv2.resize(imagearray, (28*anzahl, 28))
    return resized_image
    # breite = len(imagearray[0])
    # höhe = len(imagearray)
    #
    # k = int(np.floor(höhe/28))
    # print(np.floor(höhe/28), höhe/k)
    #
    # print("k", k)
    # anzahl = int(np.ceil(breite/höhe))
    # print("Anzahl: ", anzahl)
    # kk = int(np.floor(breite/(anzahl*28)))
    #
    # out = np.ndarray([1 + int(höhe/k), anzahl*32]).astype("uint8")
    #
    # index1 = 0
    # for i in range(0, höhe, k):
    #     index2 = 0
    #     for ii in range(0, breite, kk):
    #         out[index1][index2] = imagearray[i, ii]
    #         index2 += 1
    #     index1 += 1
    #
    # out = np.delete(out, 0, 0)
    # out = np.delete(out, -1, 0)
    # for i in range(anzahl):
    #     out = np.delete(out, -1, 1)
    # return out


def baw(imagearray):
    for i in range(imagearray.shape[0]):
        for ii in range(imagearray.shape[1]):
            if imagearray[i][ii] > 125:
                imagearray[i][ii] = 0
            else:
                imagearray[i][ii] = 255
    return imagearray

if __name__ == "__main__":
    img = Image.open("__files/zahl.jpg").convert('L')
    img = np.array(img)
    img = baw(img)

    print(img.shape)
    img = scale(img)

    # 28 * 28 = 784
    # 28 * 4 = 112
    numbers = []
    for i in range(int(img.shape[1]/28)):
        zahl = Zahl(img[:, 28*i:28*(i+1)])
        numbers.append(zahl)

    plt.imshow(numbers[0].imagearray, cmap="gray")
    # plt.show()

    a = ';'.join(str(i[0]) for i in numbers[0].imagearray.reshape(-1, 1))
    b = ';'.join(str(i[0]) for i in numbers[1].imagearray.reshape(-1, 1))
    data = {"values": [a, b, a]}
    # print(data)
    response = requests.post("http://127.0.0.1:7777/", json=data)
    df = pd.read_json(requests.get("http://127.0.0.1:7777/").text)
    print(df["values"])

    #
    # train_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.MNIST('__files', train=True, download=True,
    #                                transform=torchvision.transforms.Compose([
    #                                    torchvision.transforms.ToTensor(),
    #                                    torchvision.transforms.Normalize(
    #                                        (0.1307,), (0.3081,))
    #                                ])),
    #     batch_size=128, shuffle=True)
    # examples = enumerate(train_loader)
    # batch_idx, (example_data, example_targets) = next(examples)
    # fig = plt.figure()
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1)
    #     plt.tight_layout()
    #     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    #     plt.title("Ground Truth: {}".format(example_targets[i]))
    #     plt.xticks([])
    #     plt.yticks([])
    # # plt.show()
    #
    # #print(example_data.detach().numpy().reshape(len(example_data),-1).shape)
    # param_C = 5
    # param_gamma = 0.05 * 255
    # svm = SVC(C=param_C)  # , gamma=param_gamma)
    #
    # # svm = SVC(kernel="rbf", C=5, gamma=0.05) #LinearSVC(dual=False)
    # #svm.fit(example_data.detach().numpy().reshape(-1, 1), example_targets)
    #
    # for img, label in train_loader:
    #     # print(img.shape)
    #     # print(img.reshape(128, -1).shape)
    #     if img.shape[0] == 128:
    #         svm.fit(img.reshape(128, -1), label)
    #
    # # print(numbers[0].flat().shape)
    # pred = svm.predict(numbers[3].flat())
    # print(pred)