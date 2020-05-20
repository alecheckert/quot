#!/usr/bin/env python
"""
DetectViewer.py -- experiment with detection settings on 
a single file

"""
import sys

# File paths
import os 

# Array handling
import numpy as np 

# DataFrames
import pandas as pd 

# Read TOML format
from ..read import read_config 

# Image file reader
from ..ChunkFilter import ChunkFilter

# Core detection function
from ..spot import detect 

# Core GUI utilities
from PySide2.QtCore import Qt 
from PySide2.QtWidgets import QApplication, QWidget, QLabel, \
    QPushButton, QGridLayout, QVBoxLayout

# pyqtgraph utilities for rendering images and spots
from pyqtgraph import ImageView, ScatterPlotItem 

# Custom GUI utilities
from .guiUtils import FloatSlider, IntSlider, LabeledQComboBox, \
    set_dark_app, LabeledImageView, getTextInputs, keys_to_str

# Configuration settings for each slider
CONFIG = read_config("quot/gui/CONFIG.toml")
FILTER_SLIDER_CONFIG = CONFIG['filter_slider_config']
DETECT_SLIDER_CONFIG = CONFIG['detect_slider_config']

class DetectViewer(QWidget):
    def __init__(self, path, start_frame=0, stop_frame=100,
        gui_size=800, parent=None):
        super(DetectViewer, self).__init__(parent)
        self.path = path 
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.gui_size = gui_size

        # Load the iamge file
        self.initData()

        # Initialize the user interface
        self.initUI()

    def initData(self):
        """
        Try to load data from the specified image file path. 

        """
        # Open an image file reader
        self.ChunkFilter = ChunkFilter(self.path, start=self.start_frame,
            method_static=False, chunk_size=100)
        self.stop_frame = min(self.ChunkFilter.n_frames-1, self.stop_frame)

        # Set the four display images
        raw_image = self.ChunkFilter.get_frame(self.start_frame)
        self.images = [raw_image, raw_image.copy(), raw_image.copy(), 
            np.zeros(raw_image.shape, dtype=np.uint8)]

        # Initially the filter sliders have no identity
        self.filter_slider_ids = []

        # Initially no filtering kwargs
        self.filter_kwargs = {}

        # Figure out GUI size based on the size of 
        # the frame
        self.AR = float(self.ChunkFilter.width) / self.ChunkFilter.height 
        self.gui_height = self.gui_size 
        self.gui_width = self.gui_height * 2

    def initUI(self):

        # Master layout
        self.win = QWidget()
        L_master = QGridLayout(self.win)

        # Left subwindow, for widgets
        win_left = QWidget(self.win)
        L_left = QGridLayout(win_left)
        L_master.addWidget(win_left, 0, 0, 1, 3)

        # Right subwindow, for images
        win_right = QWidget(self.win)
        L_right = QGridLayout(win_right)
        L_master.addWidget(win_right, 0, 3, 1, 1)

        # Add four LabeledImageViews to this window
        liv_labels = ['(1) Raw', '(2) Filtered', '(3) Convolved', '(4) Binary']
        self.LIVs = [LabeledImageView(parent=win_left, label=liv_labels[j]) \
            for j in range(4)]
        for j in range(4):
            L_left.addWidget(self.LIVs[j], j%2, j//2)
            self.LIVs[j].setImage(self.images[j])

        # Link the views between the ImageViews
        for j in range(1, 4):
            self.LIVs[j].ImageView.view.setXLink(self.LIVs[0].ImageView.view)
            self.LIVs[j].ImageView.view.setYLink(self.LIVs[0].ImageView.view)       

        ## WIDGETS

        widget_align = Qt.AlignTop
        slider_min_width = 125

        # Frame slider
        self.frame_slider = IntSlider(minimum=self.start_frame, maximum=self.stop_frame,
            interval=1, name='Frame', init_value=0, min_width=slider_min_width,
            parent=win_right)
        L_right.addWidget(self.frame_slider, 0, 0, alignment=widget_align)
        self.frame_slider.assign_callback(self.frame_slider_callback)

        # Select filtering method
        filter_methods = keys_to_str(FILTER_SLIDER_CONFIG.keys())
        self.M_filter = LabeledQComboBox(filter_methods, "Filter method",
            init_value="identity", parent=win_right)
        L_right.addWidget(self.M_filter, 1, 0, alignment=widget_align)
        self.M_filter.assign_callback(self.M_filter_callback)

        # Select filtering chunk size
        chunk_size_choices = [str(i) for i in [2, 5, 10, 15, 20, 30, 40, 60, 80, 100, 200]]
        self.M_chunk_size = LabeledQComboBox(chunk_size_choices, "Chunk size")
        L_right.addWidget(self.M_chunk_size, 2, 0, alignment=widget_align)
        self.M_chunk_size.assign_callback(self.M_chunk_size_callback)
        self.M_chunk_size.setCurrentText(str(self.ChunkFilter.chunk_size))

        # Three semantically-flexible filtering FloatSliders
        self.filter_sliders = [FloatSlider(parent=win_right, 
            min_width=slider_min_width) for j in range(3)]
        for i in range(len(self.filter_sliders)):
            L_right.addWidget(self.filter_sliders[i], i+3, 0, alignment=widget_align)
            self.filter_sliders[i].assign_callback(getattr(self, "filter_slider_%d_callback" % i))
            self.filter_sliders[i].hide()

        # Button to change ROI
        self.B_change_roi = QPushButton("Change ROI", win_right)
        L_right.addWidget(self.B_change_roi, 6, 0, alignment=widget_align)
        self.B_change_roi.clicked.connect(self.B_change_roi_callback)

        # Select detection method
        detect_methods = keys_to_str(DETECT_SLIDER_CONFIG.keys())
        self.M_detect = LabeledQComboBox(detect_methods, "Detect method",
            init_value="llr", parent=win_right)
        L_right.addWidget(self.M_detect, 0, 1, alignment=widget_align)

        # Set the initial detection method and arguments
        init_detect_method = 'llr_rect'
        self.M_detect.setCurrentText(init_detect_method)
        self.detect_kwargs = {DETECT_SLIDER_CONFIG[init_detect_method][i]['name']: \
            DETECT_SLIDER_CONFIG[init_detect_method][i]['init_value'] \
                for i in DETECT_SLIDER_CONFIG[init_detect_method].keys()}
        self.detect_kwargs['method'] = init_detect_method
        self.M_detect.assign_callback(self.M_detect_callback)

        # Four semantically-flexible detection FloatSliders
        self.detect_sliders = [FloatSlider(parent=win_right,
            min_width=slider_min_width) for j in range(4)]
        for i in range(len(self.detect_sliders)):
            L_right.addWidget(self.detect_sliders[i], i+1, 1, alignment=widget_align)
            self.detect_sliders[i].assign_callback(getattr(self, "detect_slider_%d_callback" % i))

        # Autoscale the detection threshold
        self.B_auto_threshold = QPushButton("Rescale threshold", parent=win_right)
        L_right.addWidget(self.B_auto_threshold, 5, 1, alignment=widget_align)
        self.B_auto_threshold.clicked.connect(self.B_auto_threshold_callback)

        ## EMPTY WIDGETS
        for j in range(7, 24):
            _ql = QLabel(win_right)
            L_right.addWidget(_ql, j, 0, alignment=widget_align)

        ## INITIALIZATION
        self.change_detect_method(init_detect_method)
        self.filter()
        self.detect()
        self.auto_threshold()
        self.update_images(1, 2, 3, autoRange=True, autoLevels=True,
            autoHistogramRange=True)

        # Resize GUI
        self.win.resize(self.gui_width, self.gui_height)

        # Show the main window
        self.win.show()

    ########################
    ## CORE GUI FUNCTIONS ##
    ########################

    def update_images(self, *indices, autoRange=False, autoLevels=False, 
        autoHistogramRange=False):
        """
        Set the images shown in self.LIVs to the current
        values of self.images.

        args
        ----
            indices     :   list of int, including 0, 1, 2, or 3; the
                            indices of the images to update. For instance,
                            if indices = [0], then only the first image
                            is updated.
            autoRange   :   bool, reset the image limits
            autoLevels  :   bool, reset the LUT levels
            autoHistogramRange  :   bool, reset the LUT histogram range

        """
        for i in indices:
            self.LIVs[i].setImage(self.images[i], 
                autoRange=autoRange, autoLevels=autoLevels,
                autoHistogramRange=autoHistogramRange)

    def filter(self):
        """
        Filter the current frame with the current filtering settings.

        """
        self.images[1] = self.ChunkFilter.filter_frame(
            self.frame_slider.value())

    def detect(self):
        """
        Run detection on the current filtered frame with the current
        detection settings.

        """
        self.images[2], bin_img, self.detections = \
            detect(self.images[1], return_filt=True, **self.detect_kwargs)
        self.images[3] = bin_img.astype('uint8')

    def change_filter_method(self, method):
        """
        Update the sliders and other visual details for a different
        filtering method.

        """
        # Get the number of sliders required by this method
        self.filter_slider_ids = keys_to_str(FILTER_SLIDER_CONFIG[method].keys())
        n = min(len(self.filter_slider_ids), 3)

        # Reconfigure as many sliders as are needed to represent
        # this method's parameters
        self.filter_kwargs = {}
        for i, slider_id in enumerate(self.filter_slider_ids):
            self.filter_sliders[i].show()
            C = FILTER_SLIDER_CONFIG[method][slider_id]
            self.filter_sliders[i].configure(minimum=C['minimum'],
                maximum=C['maximum'], interval=C['interval'],
                return_int=(C['type']=='int'), name=C['name'])

            self.filter_kwargs[C['name']] = C['init_value']

            # Set the initial value of this slider, but don't recache yet
            # (setValueBlock() avoids recaching) since we recache at the 
            # set_method_kwargs() call if necessary. Usually recaching is 
            # only necessary if the method involves filtering the image.
            self.filter_sliders[i].setValueBlock(C['init_value'])

        # Hide the sliders that aren't needed
        for i in range(n, 3):
            self.filter_sliders[i].hide()

        # Update the filterer's kwargs
        self.ChunkFilter.set_method_kwargs(method=method, **self.filter_kwargs)

    def change_detect_method(self, method):
        """
        Update the sliders and other details for a different 
        detection method.

        """
        # Get the number of sliders required by this method
        self.detect_slider_ids = keys_to_str(DETECT_SLIDER_CONFIG[method].keys())
        n = min(len(self.detect_slider_ids), 4)

        # Reconfigure as many sliders as are needed to represent
        # this method's parameters
        self.detect_kwargs = {'method': method}
        for i, slider_id in enumerate(self.detect_slider_ids):
            self.detect_sliders[i].show()
            C = DETECT_SLIDER_CONFIG[method][slider_id]
            self.detect_sliders[i].configure(minimum=C['minimum'],
                maximum=C['maximum'], interval=C['interval'],
                return_int=(C['type']=='int'), name=C['name'])
            self.detect_kwargs[C['name']] = C['init_value']

            # Set the initial value of this slider while restraining
            # the callback from recomputing the detections
            self.detect_sliders[i].setValueBlock(C['init_value'])

        # Hide the sliders that aren't needed
        for i in range(n, 4):
            self.detect_sliders[i].hide()

    def load_frame(self, frame_index):
        """
        Change the current frame index, updating the images
        accordingly.

        args
        ----
            frame_index     :   int, the new frame index

        """
        self.images[0] = self.ChunkFilter.get_frame(frame_index)

    def auto_threshold(self):
        """
        Set the detection threshold dynamically. If the detection
        method does not have a threshold argument ("t"), then do
        nothing.

        """
        if 't' in self.detect_kwargs.keys():
            self.detect_kwargs['t'] = np.percentile(self.images[2], 99.9)
            self.images[3] = (self.images[2]>=self.detect_kwargs['t']).astype(np.uint8)

            i = self.detect_slider_ids.index('t')
            self.detect_sliders[i].configure(minimum=self.images[2].min(), 
                maximum=self.images[2].max())
            self.detect_sliders[i].setValueBlock(self.detect_kwargs['t'])

    ######################
    ## WIDGET CALLBACKS ##
    ######################

    def frame_slider_callback(self):
        """
        Execute upon a change in the value of the frame slider.

        """
        self.load_frame(self.frame_slider.value())
        self.filter()
        self.detect()
        self.update_images(0, 1, 2, 3)

    def M_filter_callback(self):
        """
        Execute upon a change in the value of M_filter,
        which selects the filtering method.

        """
        # Get the new method
        method = self.M_filter.currentText()

        # Update the sliders and the filterer object
        self.change_filter_method(method)

        # Refilter the raw image
        self.filter()

        # Run detection
        self.detect()

        # Update the plots
        self.update_images(1, 2, 3, autoRange=False, 
            autoHistogramRange=True, autoLevels=True)

    def M_chunk_size_callback(self):
        """
        Change the current filtering chunk size.

        """
        chunk_size = int(self.M_chunk_size.currentText())
        self.ChunkFilter.set_chunk_size(chunk_size)
        self.filter()
        self.detect()
        self.update_images(1, 2, 3)

    def B_change_roi_callback(self):
        """
        Execute upon clicking B_change_roi, which prompts
        the user to enter a new ROI for this file.

        """
        pass 

    def filter_slider_callback(self, i):
        """
        Respond to change in filter slider *i*.

        args
        ----
            i       :   int between 0 and 2 inclusive

        """
        # Update the corresponding filtering kwarg
        self.filter_kwargs[self.filter_slider_ids[i]] = self.filter_sliders[i].value()

        # Update the ChunkFilter object
        if 'k' in self.filter_kwargs.keys():
            self.ChunkFilter.set_method_kwargs(recache_filtered_image=True,
                **self.filter_kwargs)
        else:
            self.ChunkFilter.set_method_kwargs(**self.filter_kwargs)

        # Refilter the image
        self.filter()

        # Run detection on the new filtered image
        self.detect()

        # Update the plots
        self.update_images(1, 2, 3)

    def filter_slider_0_callback(self):
        self.filter_slider_callback(0)

    def filter_slider_1_callback(self):
        self.filter_slider_callback(1)

    def filter_slider_2_callback(self):
        self.filter_slider_callback(2)

    def M_detect_callback(self):
        """
        Respond to a change in the detection setting.

        """
        # Get the current detection method
        method = self.M_detect.currentText()

        # Update the sliders and other plot parameters
        self.change_detect_method(method)

        # Run detection
        self.detect()

        # Rescale the threshold
        self.auto_threshold()

        # Update plots
        self.update_images(2, 3, autoRange=False,
            autoLevels=True, autoHistogramRange=True)

    def detect_slider_callback(self, i):
        """
        Respond to a change in detect slider *i*.

        args
        ----
            i       :   int between 0 and 3 inclusive

        """
        # Update the corresponding detect kwarg
        self.detect_kwargs[self.detect_slider_ids[i]] = self.detect_sliders[i].value()

        # Run detection again
        self.detect()

        # Update the plots
        self.update_images(2, 3)

    def detect_slider_0_callback(self):
        self.detect_slider_callback(0)

    def detect_slider_1_callback(self):
        self.detect_slider_callback(1)

    def detect_slider_2_callback(self):
        self.detect_slider_callback(2)

    def detect_slider_3_callback(self):
        self.detect_slider_callback(3)       

    def B_auto_threshold_callback(self):
        """
        Rescale the threshold argument automatically.

        """
        self.detect()
        self.auto_threshold()
        self.update_images(2, 3, autoLevels=True, autoHistogramRange=True)

if __name__ == '__main__':
    app = QApplication()
    set_dark_app(app)
    ex = DetectViewer('78203_BioPipeline_Run1_20200508_222010_652__Plate000_WellB19_ChannelP95_Seq0035.nd2',
        start_frame=900, stop_frame=1000)
    sys.exit(app.exec_())



