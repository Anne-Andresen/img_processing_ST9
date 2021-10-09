# https://github.com/KeremTurgutlu/dicom-contour/blob/master/tutorial.ipynb
import matplotlib as mpl
import matplotlib.image
import matplotlib.pyplot as plt
import pydicom
import dicom_contour
from dicom_contour.contour import get_roi_names, cfile2pixels
from dicom_contour.plots import show_img_msk_fromarray
from pydicom.data import test_files

## Single frame 
src = 'D:Transverse_plane/MR.jPMlAcxkTWOOgufU9ffgmEr5h.Image 1.0003.dcm'
ds = pydicom.dcmread(src)
path = 'D:/rs/'
contour_file = 'RS.jPMlAcxkTWOOgufU9ffgmEr5h.AX T2.0001.dcm'
contour_data = pydicom.read_file(path + '/' + contour_file)
print(contour_data)
contour_data.dir("contour")
get_roi_names(contour_data)
print('hej')
contour_arrays = dicom_contour.contour.cfile2pixels(file='1.2.246.352.221.5267297548635971026.16569302649968573358', path='D:/rs') # Trouble some cant find path or file
print('boo')
contour_arrays[0]
first_image, first_contour, img_id = contour_arrays[4]
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(first_image)
plt.subplot(1,2,2)
plt.imshow(first_contour)
''' Slice order '''
ordered_slices = dicom_contour.contour.slice_order(path)
ordered_slices[:5]
''' Get contour dictionary '''
contour_dict = dicom_contour.contour.get_contour_dict(contour_file,path,0)
''' Getting data '''
images, contours = dicom_contour.contour.get_data(path,index=0)
images.shape, contours.shape

for img_arr, contour_arr in zip(images[58:59], contours[58:59]):
    dicom_contour.contour.plot2dcontour(img_arr, contour_arr)

cntr = contours[59]
plt.imshow(cntr)
filled_cntr = dicom_contour.contour.fill_contour(cntr)
plt.imshow(filled_cntr)

'''
['ROIContourSequence']
ctrs = contour_data.ROIContourSequence
ctrs[0]
ctrs[0].ContourSequence[0].ContourData
print(ctrs[0].ContourSequence[0].ContourData)
show_img_msk_fromarray(ds, contour_data, alpha=0.35, sz=7, cmap='inferno', save_path=None)

'''
'''
# Normal mode:
print()
print(f"File path........: {src}")
print(f"SOP Class........: {ds.SOPClassUID} ({ds.SOPClassUID.name})")
print()

pat_name = ds.PatientName
display_name = pat_name.family_name + ", " + pat_name.given_name
print(f"Patient's Name...: {display_name}")
print(f"Patient ID.......: {ds.PatientID}")
print(f"Modality.........: {ds.Modality}")
print(f"Study Date.......: {ds.StudyDate}")
#print(f"Image size.......: {ds.Rows} x {ds.Columns}")
#print(f"Pixel Spacing....: {ds.PixelSpacing}")

# use .get() if not sure the item exists, and want a default value if missing
print(f"Slice location...: {ds.get('SliceLocation', '(missing)')}")
'''
# plt.imshow(ds.pixel_array, cmap=plt.cm.get_cmap('Greys_r', 10))
# plt.imshow( ds.pixel_array[8], cmap=plt.cm.hsv)
# plt.imshow( ds.pixel_array[8])
'''
fig = plt.figure(figsize=(50, 50))  # width, height in inches

for i in range(11):
    sub = fig.add_subplot(11, 1, i + 1)
    sub.imshow(ds[0, i, :, :], interpolation='nearest')
'''
# plt.show()
# <matplotlib.image.AxesImage object at ...>


''' Otherway '''
'''
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import pydicom
import pylab as pl
import sys
import matplotlib.path as mplPath

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('Scroll to Navigate through the DICOM Image Slices')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        ax.set_ylabel('Slice Number: %s' % self.ind)
        self.im.axes.figure.canvas.draw()

fig, ax = plt.subplots(1,1)

os.system("tree C:/Users/ahaan/OneDrive/skrivebord/9.Semester/DICOM_data/")

plots = []

for f in glob.glob("C:/Users/ahaan/OneDrive/skrivebord/9.Semester/DICOM_data/0020.DCM"):
    pass
    filename = f.split("/")[-1]
    ds = pydicom.dcmread(filename)
    pix = ds.pixel_array
    pix = pix*1+(-1024)
    plots.append(pix)

y = np.dstack(plots)

tracker = IndexTracker(ax, y)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()
'''
'''
import glob
import pydicom

pixel_data = []
paths = glob.glob("C:/Users/ahaan/OneDrive/skrivebord/9.Semester/DICOM_data/*.DCM")
for path in paths:
    dataset = pydicom.dcmread(path)
    pixel_data.append(dataset.pixel_array)
    
'''

'''
dataset = pydicom.dcmread('C:/Users/ahaan/OneDrive/skrivebord/9.Semester/DICOM_data/0009.DCM')
frame_generator = pydicom.encaps.generate_pixel_data_frame(dataset.PixelData)

if 'PixelData' in dataset:
    rows = int(dataset.Rows)
    cols = int(dataset.Columns)
    print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
        rows=rows, cols=cols, size=len(dataset.PixelData)))
    if 'PixelSpacing' in dataset:
        print("Pixel spacing....:", dataset.PixelSpacing)

plt.imshow(dataset.pixel_array)
plt.show()
'''
'''
# common packages
import numpy as np
import os
import copy
from math import *
import matplotlib.pyplot as plt
from functools import reduce
# reading in dicom files
import pydicom
# skimage image processing packages
from skimage import measure, morphology
from skimage.morphology import ball, binary_closing
from skimage.measure import label, regionprops
# scipy linear algebra functions
from scipy.linalg import norm
import scipy.ndimage
# ipywidgets for some interactive plots
from ipywidgets.widgets import *
import ipywidgets as widgets
# plotly 3D interactive graphs
import plotly
from plotly.graph_objs import *

# import chart_studio.plotly as py

# set plotly credentials here
# this allows you to send results to your account plotly.tools.set_credentials_file(username=your_username, api_key=your
_key)
'''

''' 


def load_scan(path):
    slices = [pydicom.dcmread(path + '/' + s) for s in
              os.listdir(path)]
    slices = [s for s in slices if 'SliceLocation' in s]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        for s in slices:
            s.SliceThickness = slice_thickness
        return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

        image += np.int16(intercept)

        return np.array(image, dtype=np.int16)


# set path and load files
path = 'C:/Users/ahaan/OneDrive/skrivebord/9.Semester/DICOM_data/MRBRAIN.DCM'
patient_dicom = load_scan(path)
patient_pixels = get_pixels_hu(patient_dicom)
# sanity check
plt.imshow(patient_pixels[326], cmap=plt.cm.bone)
plt.show()
'''
'''
def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image >= -700, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is
    air.
    # Improvement: Pick multiple background labels from around the
    patient
    # More resistant to “trays” on which the patient lays cutting
    the
    air
    around
    the
    person in half


background_label = labels[0, 0, 0]

# Fill the air around the person
binary_image[background_label == labels] = 2

# Method of filling the lung structures (that is superior to
# something like morphological closing)
if fill_lung_structures:
    # For every slice we determine the largest solid structure
    for i, axial_slice in enumerate(binary_image):
        axial_slice = axial_slice — 1
        labeling = measure.label(axial_slice)
        l_max = largest_label_volume(labeling, bg=0)

        if l_max is not None:  # This slice contains some lung
            binary_image[i][labeling != l_max] = 1
binary_image -= 1  # Make the image actual binary
binary_image = 1 - binary_image  # Invert it, lungs are now 1

# Remove other air pockets inside body
labels = measure.label(binary_image, background=0)
l_max = largest_label_volume(labels, bg=0)
if l_max is not None:  # There are air pockets
    binary_image[labels != l_max] = 0'''
'''
    return binary_image
    
'''
