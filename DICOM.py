
import os, glob

#import pylab as pl
#import sys
# import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import pydicom

'''
class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('Scroll to Navigate through the DICOM Image Slices')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices // 2

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
        self.ax.set_ylabel('Slice Number: %s' % self.ind)
        self.im.axes.figure.canvas.draw()
'''

fig, ax = plt.subplots(1, 1)
os.system(
    "C:/Users/ahaan/OneDrive/Skrivebord/files/DICOM.py") # Use the place where code file is located on computer through python module

plots = []
f = 0
for f in glob.glob("D:/AAU Anne-Copy/Reg/"):
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



# y = numpy.dstack(plots)
print('stacked')
#tracker = IndexTracker(ax, y)
print('tracker')

# Plotting two manually
'''
plt.figure(1)
plt.subplot(211)  # total number of graphs, number of columns, placement of specific graph 
plt.imshow(plots[1])

plt.subplot(212)
plt.imshow(plots[2])
plt.show()

'''
# Plots all in seperate figures :D


i = 1
for i in range(1, len(plots)):
    plt.imshow(plots[i], cmap=plt.cm.get_cmap('Greys_r', 10))
    plt.xlabel('Transversal')
    plt.ylabel('  ')
    plt.axis('off')
    plt.title('Slice %d' %i)
    plt.show()






'''
def plot3d(image):
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, image)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
'''
'''
if __name__ == "__main__":
    # mg = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
      #               [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
       #              [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
    ds = pydicom.dcmread("D:Transverse_plane/MR.jPMlAcxkTWOOgufU9ffgmEr5h.Image 3.0001.dcm")
    img = ds.pixel_array
plot3d(img)
'''