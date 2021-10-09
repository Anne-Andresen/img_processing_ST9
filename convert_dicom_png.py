import numpy as np
import pydicom
from PIL import ImageTk, Image
import os, glob
# Works for a single image at a time

plots = []
f = 1
'''
for f in glob.glob("D:Transverse_plane/MR.jPMlAcxkTWOOgufU9ffgmEr5h.Image*.0001.dcm"):
    pass
    filename = f.split("/")[-1]
    print(filename)
    ds = pydicom.dcmread(filename)
    print('read')
    pix = ds.pixel_array
    # pix = pix*1+(-1024)
    print('pic')
    plots.append(pix)
    print('appended')
'''
i=1
for f in glob.glob("D:/Patient_1/study_1/1/*.dcm"):
    # filename = f.split("/")[-1]
    # print(filename)
    filename = f
    file = os.path.basename(filename)
    y = os.path.splitext(file)[0]
    ds = pydicom.dcmread(filename) # Put the right path of the image that you want to convert
    new_image = ds.pixel_array.astype(float) # Convert the values into float
    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
    scaled_image = np.uint8(scaled_image)
    final_image = Image.fromarray(scaled_image)
    i=i+1

    final_image.save('D:/Patient_1/study_1/FN1/' + y + '.png') # Need to work on getting the right names
print('DONE')
#final_image.show() # Display the Image

# final_image.save('image.jpg') # Save the image as JPG
#final_image.save('D:output/image.png') # Save the image as PNG
