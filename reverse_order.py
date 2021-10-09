import numpy as np
import cv2 as cv
import os
import glob

img = []
fn = []



for f in glob.glob('C:/Users/ahaan/OneDrive/Skrivebord/input_patches/GTV-N-MR1/*.png'):
    filename = f
    pic = cv.imread(filename)
    img.append(pic)
    file = os.path.basename(filename)
    y = os.path.splitext(file)[0]
    fn.append(y)

print('Array images:', img)
print('Array filenames:', fn)

rever = fn[::-1]
print('reversed list ', rever)

# Save image from array with reversed names.
path = 'C:/Users/ahaan/OneDrive/Skrivebord/input_patches/reverse/'
for i in range(0, 65):
    cv.imwrite(path + rever[i] + '.png', img[i])
    print('saved')


