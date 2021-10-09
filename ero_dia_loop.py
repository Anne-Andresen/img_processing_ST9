import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def dilation(img, kernel_size, iteration):
    im = cv.imread(img)
    x = np.ones((kernel_size, kernel_size), np.uint8)
    z = iteration
    dila_img = cv.dilate(im, kernel=x, iterations=z)
    print('dilated')
    return dila_img


def erosion(img, kernel_size, iteration):
    im = cv.imread(img)
    x = np.ones((kernel_size, kernel_size), np.uint8)
    z = iteration
    eroded_img = cv.erode(im, kernel=x, iterations=z)
    return eroded_img


for f in glob.glob('D:/Patient-1/study_2/Intensity_normalization_2/000*.png'):
    filename = f
    dilate = dilation(filename, 5, 1)
    file = os.path.basename(filename)
    y = os.path.splitext(file)[0]
    print('dilated:', y)
    path = 'D:/Patient-1/study_2/dilated/'
    cv.imwrite(path + y + '.png', dilate)

for f in glob.glob('D:/Patient-1/study_2/Intensity_normalization_2/000*.png'):
    filename = f
    erode = erosion(filename, 5, 1)
    file = os.path.basename(filename)
    y = os.path.splitext(file)[0]
    print('eroded:', y)
    path = 'D:/Patient-1/study_2/eroded/'
    cv.imwrite(path + y + '.png', erode)

