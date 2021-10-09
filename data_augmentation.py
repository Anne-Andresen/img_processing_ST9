# Translation, rotation, scaling(erosion dilation)  and mirroring.
import numbers
import warnings

import cv2
import cv2 as cv
import PIL
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import random
import scipy
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import zoom


def rotation(img, gt):
    low = 1
    high = 359
    degree = random.randint(low, high)
    img = Image.open(img)
    gt = Image.open(gt)
    rotation_img = rotate(img, degree, reshape=False)
    rotation_gt = rotate(gt, degree, reshape=False)

    return rotation_img, rotation_gt


def zoom(img, gt, zoom_coeff,  **kwargs):
    img = Image.open(img)
    gt = Image.open(gt)
    h, w = img.shape[:2]
    h1, w1 = gt.shape[:2]
    zoom_tuple = (zoom_coeff,) * 2 + (1,) * (img.ndim - 2)
    zoom_tuple1 = (zoom_coeff,) * 2 + (1,) * (gt.ndim - 2)
    # Zoom out
    if zoom_coeff < 1:
        zh = int(np.round(h * zoom_coeff))
        zw = int(np.round(w * zoom_coeff))
        top = (h - zh) // 2
        left = (w - zw) // 2
        zh1 = int(np.round(h1 * zoom_coeff))
        zw1 = int(np.round(w1 * zoom_coeff))
        top1 = (h1 - zh1) // 2
        left1 = (w - zw1) // 2

        # Zero padding
        out = np.zeros(img)
        out1 = np.zeros(gt)
        out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, **kwargs)
        out1[top1:top1 + zh1, left1:left1 + zw1] = zoom(gt, zoom_tuple1, **kwargs)
    # Zoom in
    elif zoom_coeff > 1:
        zh = int(np.round(h / zoom_coeff))
        zw = int(np.round(w / zoom_coeff))
        top = (h - zh) // 2
        left = (w - zw) // 2
        zh1 = int(np.round(h1 / zoom_coeff))
        zw1 = int(np.round(w1 / zoom_coeff))
        top1 = (h1 - zh1) // 2
        left1 = (w1 - zw1) // 2

        out = zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)
        out1 = zoom(gt[top1:top1 + zh1, left1:left1 + zw1], zoom_tuple1, **kwargs)

        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        trim_top1 = ((out1.shape[0] - h1) // 2)
        trim_left1 = ((out1.shape[1] - w1) // 2)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]
        out1 = out1[trim_top1:trim_top1 + h, trim_left1:trim_left1 + w]

    else:  # If zoom_coeff ==1
        out = img
        out1 = gt

    return out, out1


def transpose(img, gt):
    img = cv.imread(img)
    gt = cv.imread(gt)
    vertical_flip = img.transpose(method=Image.FLIP_TOP_BOTTOM)
    horizontal_flip = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    vertical_flip_gt = gt.transpose(method=Image.FLIP_TOP_BOTTOM)
    horizontal_flip_gt = gt.transpose(method=Image.FLIP_LEFT_RIGHT)

    return vertical_flip, horizontal_flip, vertical_flip_gt, horizontal_flip_gt


def saturation(img, saturation):
    img = cv.cvtColor(img, cv2.COLOR_HSV2BGR)
    v = img[:, :, 2]
    v = np.where(v < 255 - saturation.v + saturation, 255)
    img[:, :, 2] = v

    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    cv.imwrite(path + 'saturation' + str(saturation) + Extension, img)


def hue(img, saturation):
    img = cv.cvtColor(img, cv2.COLOR_HSV2BGR)
    v = img[:, :, 2]
    v = np.where(v < 255 + saturation.v - saturation, 255)
    img[:, :, 2] = v

    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    cv.imwrite(path + 'saturation' + str(saturation) + Extension, img)


def bright_img(img):
    img = Image.open(img)
    ench = ImageEnhance.Brightness(img).enhance(1.5)
    #   ImageEnhance.Brightness(img).enhance(1.5).save(path + '' % img.translate(None, '.png') + '.png', "PNG")
    return ench


def blr_img(img):
    # file_path = path + img
    img = Image.open(img)
    BLR = img.filter(ImageFilter.BLUR)
    #  img.filter(ImageFilter.BLUR).save(path + '' % img.translate(None, '.png') + '.png', "PNG")
    return BLR


def contrast_enh_img(img):
    img = Image.open(img)
    CE = ImageEnhance.Contrast(img).enhance(0.8)
    # ImageEnhance.Contrast(img).enhance(0.8).save(path)
    return CE
