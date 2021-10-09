import cv2 as cv
import glob
import os
def resize(img, w, h):
    image = cv.imread(img, cv.IMREAD_UNCHANGED)
    print('original dim:', image.shape)
    dim = (w, h)
    print('bob')
    resized = cv.resize(image, dim, interpolation=cv.INTER_NEAREST)
    return resized


#im = 'C:/Users/ahaan/OneDrive/Skrivebord/input_patches/4.png'
for f in glob.glob('C:/Users/ahaan/OneDrive/Skrivebord/gt/mr1/slice*.png'):
    im = f
    ref = resize(img=im, w=640, h=640)
    print('Resized dim:', ref.shape)
    filename = im
    file = os.path.basename(filename)
    y = os.path.splitext(file)[0]
    path = 'C:/Users/ahaan/OneDrive/Skrivebord/gt/mr1/crop/'
    cv.imwrite(path + y + '.png', ref)

print('DONE')