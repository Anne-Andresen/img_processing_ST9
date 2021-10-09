import glob
import pandas as pd
import numpy as np
import cv2 as cv
import os




f=1
for f in glob.glob("D:/Patient-1/study_5/resize_5/000*.png"):
    im_path = f
    print(f)
    img = cv.imread(im_path)
    img = cv.resize(img, (640, 640))
    cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
    filename = im_path
    file = os.path.basename(filename)
    y = os.path.splitext(file)[0]
    cv.imwrite('D:/Patient-1/study_4/Intensity_normalization_5/' + y + '.png', img)


