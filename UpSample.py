import cv2 as cv


def resize(img, w, h):
    image = cv.imread(img, cv.IMREAD_UNCHANGED)
    print('original dim:', image.shape)
    dim = (w, h)
    print('bob')
    resized = cv.resize(image, dim, interpolation=cv.INTER_NEAREST)
    return resized


im = 'C:/Users/ahaan/OneDrive/Skrivebord/input_patches/4_gt.png'
ref = resize(img=im, w=1024, h=1024)
print('Resized dim:', ref.shape)
filename = 'Resized_gt.png'
path = 'C:/Users/ahaan/OneDrive/Skrivebord/input_patches/' + filename
cv.imwrite(path, ref)
