#!/usr/bin/env python
"""
masker.py -- 

"""
import sys 

# File paths
import os 

# Numeric
import numpy as np 

# Dataframes
import pandas as pd 

# File readers
from ..read import ImageReader 

# Core GUI utilities
from PySide2 import QtCore
from PySide2.QtCore import Qt 
from PySide2.QtWidgets import QWidget, QLabel, QPushButton, \
    QVBoxLayout, QGridLayout

# pyqtgraph plotting utilities
from pyqtgraph import ImageView, ScatterPlotItem, PolyLineROI

# Custom GUI utilities
from .guiUtils import IntSlider 

class Masker(QWidget):
    def __init__(self, image_path, max_points_freestyle=20, parent=None):
        super(Masker, self).__init__(parent=parent)
        self.image_path = image_path 
        self.max_points_freestyle = max_points_freestyle
        self.initData()
        self.initUI()

    def initData(self):
        """
        Load the images and related data.

        """
        # Create the image reader
        self.imageReader = ImageReader(self.image_path)

        # Get the first image
        self.image = self.imageReader.get_frame(0)

    def initUI(self):
        """
        Initialize the user interface.

        """
        # Main layout
        self.win = QWidget()
        L_master = QGridLayout(self.win)

        # An ImageView on the left to contain the subject of masking
        self.imageView = ImageView(parent=self.win)
        L_master.addWidget(self.imageView, 0, 0, 1, 2)
        self.imageView.setImage(self.image)

        # Override the default ImageView.imageItem.mouseClickEvent
        self.imageView.imageItem.mouseClickEvent = self._imageView_mouseClickEvent

        # Override the default ImageView.imageItem.mouseDoubleClickEvent 
        self.imageView.imageItem.mouseDoubleClickEvent = \
            self._imageView_mouseDoubleClickEvent 

        # All currently clicked points
        self.points = []

        # All current PolyLineROI objects
        self.polyLineROIs = []

        # ScatterPlotItem, to show current accumulated clicks
        self.scatterPlotItem = ScatterPlotItem()
        self.scatterPlotItem.setParentItem(self.imageView.imageItem)

        ## WIDGETS
        widget_align = Qt.AlignTop

        # Frame slider
        self.frame_slider = IntSlider(
            minimum=0,
            maximum=self.imageReader.n_frames-1,
            interval=1,
            init_value=0,
            name='Frame',
            parent=self.win
        )
        L_master.addWidget(self.frame_slider, 0, 2, 1, 1,
            alignment=widget_align)
        self.frame_slider.assign_callback(self.frame_slider_callback)

        # Button: create new ROI
        self.create_roi_mode = False 
        self.B_create_ROI = QPushButton("Draw ROI", parent=self.win)
        self.B_create_ROI.clicked.connect(self.B_create_ROI_callback)
        L_master.addWidget(self.B_create_ROI, 1, 2, 1, 1, 
            alignment=widget_align)

        # Button: freestyle drawing to make ROI
        self.freestyle_mode = False 
        self.B_freestyle = QPushButton("Freestyle", parent=self.win)
        self.B_freestyle.clicked.connect(self.B_freestyle_callback)
        L_master.addWidget(self.B_freestyle, 2, 2, 1, 1, 
            alignment=widget_align)

        # Resize and launch
        self.win.resize(800, 600)
        self.win.show()

    ## CORE FUNCTIONS

    def update_image(self, frame_index=None, autoRange=False, 
        autoLevels=False, autoHistogramRange=False):
        if frame_index is None:
            frame_index = self.frame_slider.value()
        self.image = self.imageReader.get_frame(frame_index)
        self.imageView.setImage(self.image, autoRange=autoRange,
            autoLevels=autoLevels, autoHistogramRange=autoHistogramRange)

    def update_scatter(self):
        """
        Write the contents of self.points to self.scatterPlotItem.

        """
        self.scatterPlotItem.setData(
            pos=np.asarray(self.points),
            pxMode=False,
            symbol='o',
            pen={'color': '#FFFFFF', 'width': 3.0},
            size=2.0,
            brush=None,
        )

    def clear_scatter(self):
        self.points = []
        self.scatterPlotItem.setData()

    def createPolyLineROI(self, points):
        p = PolyLineROI(self.points, closed=True, removable=True)
        self.polyLineROIs.append(p)
        self.imageView.view.addItem(self.polyLineROIs[-1])

        # If the user requests to remove this ROI, remove it
        self.polyLineROIs[-1].sigRemoveRequested.connect(self._remove_ROI)

    def getPoints(self, polyLineROI):
        """
        Return the set of points that make up a PolyLineROI as a 
        2D ndarray (shape (n_points, 2)).

        """
        state = polyLineROI.getState()
        return np.asarray([[p.x(), p.y()] for p in state['points']])

    ## SIGNAL CALLBACKS

    def _imageView_mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            if self.imageView.imageItem.raiseContextMenu(ev):
                ev.accept()
        if ev.button() == QtCore.Qt.LeftButton and self.create_roi_mode:
            self.points.append(ev.pos())
            self.update_scatter()

    def _imageView_mouseDoubleClickEvent(self, ev):
        if self.create_roi_mode:
            self.createPolyLineROI(self.points)
            self.clear_scatter()
            self.create_roi_mode = False 
            self.imageView.view.state['mouseEnabled'] = np.array([True, True])
        elif self.freestyle_mode:
            self.B_freestyle_callback()
        else:
            pass 

    def _remove_ROI(self, polylineroi):
        polylineroi.sigRemoveRequested.disconnect(self._remove_ROI)
        self.polyLineROIs.remove(polylineroi)
        self.imageView.view.removeItem(polylineroi)
        del polylineroi 

    ## WIDGET CALLBACKS

    def frame_slider_callback(self):
        frame_index = self.frame_slider.value()
        self.update_image(frame_index=frame_index)

    def B_create_ROI_callback(self):
        self.create_roi_mode = True
        self.imageView.view.state['mouseEnabled'] = np.array([False, False])

    def B_freestyle_callback(self):

        # End freestyle mode by creating a mask from the drawn points
        if self.freestyle_mode:
            self.freestyle_mode = False 
            self.imageView.imageItem.setDrawKernel(kernel=None,
                mask=None, center=None)
            mask = (self.image == self.draw_val)
            self.points = get_ordered_mask_points(mask, 
                max_points=self.max_points_freestyle)
            self.createPolyLineROI(self.points)
            self.frame_slider_callback()
            self.points = []
            self.B_freestyle.setText("Freestyle")

        # Start freestyle mode by enabling drawing on self.imageView.imageItem
        else:
            self.freestyle_mode = True 
            self.draw_val = int(self.image.max() + 2)
            kernel = np.array([[self.draw_val]])
            self.imageView.imageItem.setDrawKernel(kernel=kernel, 
                mask=None, center=(0,0))
            self.B_freestyle.setText("Finish freestyle")

def get_ordered_mask_points(mask, max_points=100):
    """
    Given the edges of a two-dimensional binary mask, construct a line
    around the mask.

    args
    ----
        mask        :   2D ndarray, dtype bool
        max_points  :   int, the maximum number of points tolerated
                        in the final mask. If the number of points 
                        exceeds this, the points are repeatedly 
                        downsampled until there are fewer than 
                        max_points.

    returns
    -------
        2D ndarray of shape (n_points, 2), the points belonging
            to this ROI

    """
    # Get the X and Y coordinates of all points in the mask edge
    points = np.asarray(mask.nonzero()).T

    # Keep track of which points we've included so far
    included = np.zeros(points.shape[0], dtype=np.bool)

    # Start at the first point
    ordered_points = np.zeros(points.shape, dtype=points.dtype)
    ordered_points[0,:] = points[0,:]
    included[0] = True 

    # Index of the current point
    c = 0
    midx = 0

    # Find the closest point to the current point
    while c < points.shape[0]-1:

        # Compute distances to every other point
        distances = np.sqrt(((points[midx,:]-points)**2).sum(axis=1))

        # Set included points to impossible distances
        distances[included] = np.inf 

        # Among the points not yet included in *ordered_points*,
        # choose the one closest to the current point
        midx = np.argmin(distances)

        # Add this point to the set of ordered points
        ordered_points[c+1,:] = points[midx, :]

        # Mark this point as included
        included[midx] = True 

        # Increment the current point counter
        c += 1

    # Downsample until there are fewer than max_points
    while ordered_points.shape[0] > max_points:
        ordered_points = ordered_points[::2,:]

    return ordered_points


if __name__ == '__main__':
    app = QApplication([])
    set_dark_app(app)
    q = Masker('region_6.nd2')
    sys.exit(app.exec_())
