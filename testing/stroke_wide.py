from PIL import Image
import numpy as np
import cv2

from dashboard.scanning import addBorder


def stroking(imagepath):
    #img = Image.open("/Users/jannikobenhoff/Documents/pythonProjects/quantum_computation/dashboard/__files/4_546.jpg")
    img = Image.open(imagepath)
    img = np.array(img)
    img = addBorder(img, reverse=True)
    '''255: weiß, 0: schwarz'''
    for i in range(img.shape[0]-2):
        for ii in range(img.shape[1]-2):
            if img[i, ii] > 250 and img[i, ii+1] < 40:
                img[i, ii] = 0
                img[i, ii+1] = 0
                img[i, ii+2] = 0
                img[i, ii+3] = 0
                img[i, ii-3] = 0
                img[i, ii-2] = 0
                img[i, ii-1] = 0

    for i in range(img.shape[0]-2):
        for ii in range(img.shape[0]-2):
            if img[ii, i] > 250 and img[ii+1, i] < 40:
                img[ii, i] = 0
                img[ii-1, i] = 0
                img[ii-2, i] = 0
                img[ii-3, i] = 0
                img[ii+1, i] = 0
                img[ii+2, i] = 0
                img[ii+3, i] = 0

    #img = addBorder(img, reverse=True)

    img = cv2.resize(img, (28, 28))
    # img[img > 0] = 255
    img = Image.fromarray(img)

    img.save(imagepath)


# stroking("=_444.jpg")