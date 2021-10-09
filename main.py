from data_augmentation import bright_img
from data_augmentation import blr_img
from data_augmentation import transpose
from data_augmentation import zoom
from data_augmentation import saturation
from data_augmentation import rotation
from data_augmentation import contrast_enh_img
import cv2 as cv
import glob
import os
from PIL import Image
from PIL import ImageEnhance
import matplotlib.pyplot as plt
# Brightness all img
'''
for f in glob.glob('C:/Users/ahaan/OneDrive/Skrivebord/input_patches/64/000*.png'):
    im = f
    ref = bright_img(im)
    print('brightened')
    filename = im
    file = os.path.basename(filename)
    y = os.path.splitext(file)[0]
    path = 'C:/Users/ahaan/OneDrive/Skrivebord/input_patches/64/enhanced/'
    ref.save(path + y + '.png')


for f in glob.glob('C:/Users/ahaan/OneDrive/Skrivebord/input_patches/64/000*.png'):
    im = f
    ref = blr_img(im)
    print('blurred')
    filename = im
    file = os.path.basename(filename)
    y = os.path.splitext(file)[0]
    path = 'C:/Users/ahaan/OneDrive/Skrivebord/input_patches/64/BLR/'
    ref.save(path + y + '.png')

for f in glob.glob('C:/Users/ahaan/OneDrive/Skrivebord/input_patches/64/000*.png'):
    im = f
    ref = contrast_enh_img(im)
    print('contrast enhanced')
    filename = im
    file = os.path.basename(filename)
    y = os.path.splitext(file)[0]
    path = 'C:/Users/ahaan/OneDrive/Skrivebord/input_patches/64/contrast_enhanced/'
    ref.save(path + y + '.png')
good so far'''

for f in glob.glob('C:/Users/ahaan/OneDrive/Skrivebord/input_patches/64/000*.png'):
    for z in glob.glob('C:/Users/ahaan/OneDrive/Skrivebord/input_patches/64gt/slice_*.png'):
        im = f
        gt = z
        ref, gt = rotation(im, gt)
        print('Rotated:')
        filename = im
        file = os.path.basename(filename)
        y = os.path.splitext(file)[0]
        filename1 = z
        file1 = os.path.basename(filename1)
        x = os.path.splitext(file1)[0]
        path = 'C:/Users/ahaan/OneDrive/Skrivebord/input_patches/64/rotated/'
        path1 = 'C:/Users/ahaan/OneDrive/Skrivebord/input_patches/64gt/rotated/'
        cv.imwrite(path + y + '.png', ref)
        cv.imwrite(path1 + x + '.png', gt)
        # ref.save(path + y + '.png')
        # gt.save(path1 + x + '.png')


''' works 
pic = bright_img('00049_64_128.png')
plt.imshow(pic)
plt.show()
'''
