import cv2 as cv
import glob
import os


def crop_square(img, x, y):
    image = cv.imread(img, cv.IMREAD_UNCHANGED)
    y1 = y
    y2 = y + 256
    x1 = x
    x2 = x + 256
    cropped_img = image[x1:x2, y1:y2]
    print('cropped', cropped_img.shape)
    return cropped_img


def crop_other_dim(img, x1, x2, y1, y2):
    image = cv.imread(img, cv.IMREAD_UNCHANGED)
    y1 = y1
    y2 = y2
    x1 = x1
    x2 = x2
    cropped_image = image[y1:y2, x1:x2]
    print('cropped image dim:', cropped_image.shape)
    return cropped_image

# Run crop square for multiple images and save with original names
for f in glob.glob('C:/Users/ahaan/OneDrive/Skrivebord/gt/mr6/crop/slice_*.png'):
    filename = f
    crop = crop_square(filename, 192, 192)
    file = os.path.basename(filename)
    y = os.path.splitext(file)[0]
    path = 'D:/gt_cropped/gt6/'
    cv.imwrite(path + y + '.png', crop)
'''
for f in glob.glob('D:/Patient-1/study_1/dilated/000*.png'):
    filename = f
    crop = crop_square(filename, 256, 256)
    file = os.path.basename(filename)
    y = os.path.splitext(file)[0]
    path = 'D:/Patient-1/study_1/crop_1/'
    cv.imwrite(path + y + '.png', crop)

'''
# Run crop according to specified dim for multiple images and save with original names
'''
for f in glob.glob('D:/Patient_1/study_2/FN2/MR.jPMlAcxkTWOOgufU9ffgmEr5h.Image *.png'):
    filename = f
    crop = crop_other_dim(filename, 312, 712 ,312, 712)
    file = os.path.basename(filename)
    y = os.path.splitext(file)[0]
    path = 'D:/Patient_1/study_2/FN2/cropped/'
    cv.imwrite(path + y + '.png', crop)
    '''
