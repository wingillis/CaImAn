import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import cv2
import scipy

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print(1)
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')

#%%

def make_color_img(img, gain=255, out_type=np.uint8):
    min = img.min()
    max = img.max()
    img = (img-min)/(max-min)*gain
    img = img.astype(out_type)
    img = np.dstack([img]*3)
    return img

def draw_contours():
    global thrshcomp_line, selected_components, img, background_image
    contours = [cv2.findContours(cv2.threshold(img, np.int(thrshcomp_line.value()), 255, 0)[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1] for img in components[selected_components]]
    bkgr_contours = background_image.copy()
    cv2.drawContours(bkgr_contours, sum(contours[0::6], []), -1, (255, 0, 0), 1)
    cv2.drawContours(bkgr_contours, sum(contours[1::6], []), -1, (0, 255, 0), 1)
    cv2.drawContours(bkgr_contours, sum(contours[2::6], []), -1, (0, 0, 255), 1)
    cv2.drawContours(bkgr_contours, sum(contours[3::6], []), -1, (255, 255, 0), 1)
    cv2.drawContours(bkgr_contours, sum(contours[4::6], []), -1, (255, 0, 255), 1)
    cv2.drawContours(bkgr_contours, sum(contours[5::6], []), -1, (0, 255, 255), 1)
    img.setImage(bkgr_contours, autoLevels=False)


# load estimates and movie image
mov = cm.load('/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE/YST/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap')
Cn = cm.load('/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE/YST/projections/correlation_image.tif')
background_image = make_color_img(Cn)
cnm_obj = load_CNMF('/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/WEBSITE/YST/Yr_d1_200_d2_256_d3_1_order_C_frames_3000__cnmf_perf_web_after_analysis.hdf5')
estimates = cnm_obj.estimates
estimates.restore_discarded_components()
estimates.img_components = estimates.A.toarray().reshape((estimates.dims[0], estimates.dims[1],-1), order='F').transpose([2,0,1])
estimates.cms = np.array([scipy.ndimage.measurements.center_of_mass(comp) for comp in estimates.img_components])
selected_components = np.arange(estimates.nr)


# Interpret image data as row-major instead of col-major
# pg.setConfigOptions(imageAxisOrder='row-major')

pg.mkQApp()
win = pg.GraphicsLayoutWidget()
win.setWindowTitle('pyqtgraph example: Image Analysis')

# A plot area (ViewBox + axes) for displaying the image
p1 = win.addPlot(title="Image here")


# Item for displaying image data
img = pg.ImageItem()
p1.addItem(img)

# Custom ROI for selecting an image region

roi = pg.ROI([-8, 14], [6, 5])
roi.addScaleHandle([0.5, 1], [0.5, 0.5])
roi.addScaleHandle([0, 0.5], [0.5, 0.5])
p1.addItem(roi)
roi.setZValue(10)  # make sure ROI is drawn above image


# Contrast/color control
hist = pg.HistogramLUTItem()
hist.setImageItem(img)
win.addItem(hist)

hist_cnn = pg.HistogramLUTItem()
img_cnn = pg.ImageItem()
img_cnn.setImage(np.atleast_2d(estimates.cnn_preds), autoLevels=False)
hist_cnn.setImageItem(img_cnn)
win.addItem(hist_cnn)



def changed_quality_metrics(event):
    global selected_components
    low_cnn, high_cnn = event.getLevels()
    selected_components = np.where(estimates.cnn_preds >= low_cnn)[0]
    discarded_components = np.where(estimates.cnn_preds <= low_cnn)[0]
    selected_components = np.setdiff1d(selected_components, discarded_components)
    draw_contours()


hist_cnn.sigLevelsChanged.connect(test)




# Draggable line for setting isocurve level
thrshcomp_line = pg.InfiniteLine(angle=0, movable=True, pen='g')
hist.vb.addItem(thrshcomp_line)
hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
thrshcomp_line.setValue(100)
thrshcomp_line.setZValue(1000) # bring iso line above contrast controls


# Another plot area for displaying ROI data
win.nextRow()
p2 = win.addPlot(colspan=2)
p2.setMaximumHeight(250)
win.resize(800, 800)
win.show()


# Generate image data
components /= components.max(axis=(1,2))[:,None,None]
components *= 255
components = components.astype(np.uint8)


draw_contours()
hist.setLevels(background_image.min(), background_image.max())


# set position and scale of image
img.scale(1, 1)
# img.translate(-50, 0)

# zoom to fit imageo
p1.autoRange()


# # Callbacks for handling user interaction
# def updatePlot(event):
#     global img, roi, background_image, p2
#     if event.isExit():
#         p1.setTitle("")
#         return
#     pos = event.pos()
#     i, j = pos.y(), pos.x()
#     i = int(np.clip(i, 0, background_image.shape[0] - 1))
#     j = int(np.clip(j, 0, background_image.shape[1] - 1))
#     val = background_image[i, j, 0]
#     ppos = img.mapToParent(pos)
#     x, y = ppos.x(), ppos.y()
#     p1.setTitle("pos: (%0.1f, %0.1f)  pixel: (%d, %d)  value: %g" % (x, y, i, j, val))
#
#
#     # selected = roi.getArrayRegion(data, img)
#     # p2.plot(selected.mean(axis=0), clear=True)
#
# roi.sigRegionChanged.connect(updatePlot)
# updatePlot()

thrshcomp_line.sigDragged.connect(draw_contours)


def imageHoverEvent(event):
    """Show the position, pixel, and value under the mouse cursor.
    """
    global x,y,i,j,val
    pos = event.pos()
    i, j = pos.y(), pos.x()
    i = int(np.clip(i, 0, background_image.shape[0] - 1))
    j = int(np.clip(j, 0, background_image.shape[1] - 1))
    val = background_image[i, j, 0]
    ppos = img.mapToParent(pos)
    x, y = ppos.x(), ppos.y()


# Monkey-patch the image to use our custom hover function.
# This is generally discouraged (you should subclass ImageItem instead),
# but it works for a very simple use like this.
img.hoverEvent = imageHoverEvent

def mouseClickEvent(event):
    global x, y, i, j, val
    distances = np.sum(((x,y)-cms)**2, axis=1)**0.5
    closest_component = np.argmin(distances)
    p2.plot(estimates.C[closest_component], clear=True)
    p1.setTitle("pos: (%0.1f, %0.1f)  component: %d  value: %g dist:%f" % (x, y, closest_component, val,distances[closest_component]))

p1.vb.mouseClickEvent = mouseClickEvent


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

