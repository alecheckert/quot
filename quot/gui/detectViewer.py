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
from ..chunkFilter import ChunkFilter

# Core detection function
from ..findSpots import detect 

# Core GUI utilities
import PySide2
from PySide2.QtCore import Qt 
from PySide2.QtWidgets import QApplication, QWidget, QLabel, \
    QPushButton, QGridLayout, QVBoxLayout, QDialog, QShortcut
from PySide2.QtGui import Qt as QtGui_Qt
from PySide2.QtGui import QKeySequence

# pyqtgraph utilities for rendering images and spots
from pyqtgraph import ImageView, ScatterPlotItem 
from pyqtgraph.graphicsItems import ScatterPlotItem as SPI_base

# Custom GUI utilities
from .guiUtils import FloatSlider, IntSlider, LabeledQComboBox, \
    set_dark_app, LabeledImageView, getTextInputs, keys_to_str, \
    Symbols, ROISelectionBox, MASTER_COLOR, PromptSelectROI, \
    ImageSubpositionWindow

# Configuration settings for each slider
config_path = os.path.join(os.path.dirname(__file__), "CONFIG.toml")
CONFIG = read_config(config_path)
FILTER_SLIDER_CONFIG = CONFIG['filter_slider_config']
DETECT_SLIDER_CONFIG = CONFIG['detect_slider_config']

# Customize the spot overlay symbols a little
SPI_base.Symbols['+'] = Symbols['alt +']
SPI_base.Symbols['open +'] = Symbols['open +']
symbol_sizes = {'+': 16.0, 'o': 12.0, 'open +': 16.0}

class DetectViewer(QWidget):
    def __init__(self, path, start_frame=0, stop_frame=100,
        gui_size=900, parent=None, **subregion_kwargs):
        super(DetectViewer, self).__init__(parent)
        self.path = path 
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.gui_size = gui_size

        # Load the iamge file
        self.initData(**subregion_kwargs)

        # Initialize the user interface
        self.initUI()

    def initData(self, **subregion_kwargs):
        """
        Try to load data from the specified image file path. 

        """
        # Open an image file reader
        self.ChunkFilter = ChunkFilter(self.path, start=self.start_frame,
            method_static=False, chunk_size=100, init_load=False)
        self.stop_frame = min(self.ChunkFilter.n_frames-1, self.stop_frame)
        self.ChunkFilter.set_subregion(**subregion_kwargs)
        self.ChunkFilter.load_chunk(self.start_frame)

        # Set the four display images
        raw_image = self.ChunkFilter.get_subregion(self.start_frame, \
            **self.ChunkFilter.sub_kwargs)
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
        self.gui_width = self.gui_height * 1.75

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

        # Set up three ScatterPlotItems for each of the first three ImageViews,
        # for overlaying detections as symbols
        self.SPIs = [ScatterPlotItem() for j in range(3)]
        for i in range(3):
            self.SPIs[i].setParentItem(self.LIVs[i].ImageView.imageItem)

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

        # Button to change frame range
        self.B_change_frame_range = QPushButton("Change frame range", win_right)
        L_right.addWidget(self.B_change_frame_range, 7, 0, alignment=widget_align)
        self.B_change_frame_range.clicked.connect(self.B_change_frame_range_callback)

        # Select detection method
        detect_methods = keys_to_str(DETECT_SLIDER_CONFIG.keys())
        self.M_detect = LabeledQComboBox(detect_methods, "Detect method",
            init_value="llr", parent=win_right)
        L_right.addWidget(self.M_detect, 0, 1, alignment=widget_align)

        # Set the initial detection method and arguments
        init_detect_method = 'llr'
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
        L_right.addWidget(self.B_auto_threshold, 6, 1, alignment=widget_align)
        self.B_auto_threshold.clicked.connect(self.B_auto_threshold_callback)

        # Overlay detections as symbols
        self.B_spot_overlay_state = False 
        self.B_spot_overlay = QPushButton("Overlay detections", parent=win_right)
        L_right.addWidget(self.B_spot_overlay, 7, 1, alignment=widget_align)
        self.B_spot_overlay.clicked.connect(self.B_spot_overlay_callback)

        # Change the symbol used for spot overlays
        symbol_choices = ['+', 'o', 'open +']
        self.M_symbol = LabeledQComboBox(symbol_choices, "Spot symbol",
            init_value='o', parent=win_right)
        L_right.addWidget(self.M_symbol, 5, 1, alignment=widget_align)
        self.M_symbol.assign_callback(self.M_symbol_callback)

        # Save result to file
        self.B_save = QPushButton("Save settings", parent=win_right)
        L_right.addWidget(self.B_save, 8, 1, alignment=widget_align)
        self.B_save.clicked.connect(self.B_save_callback)

        # Show individual spots
        self.B_show_spots = QPushButton("Show individual spots", parent=win_right)
        L_right.addWidget(self.B_show_spots, 9, 1, alignment=widget_align)
        self.B_show_spots.clicked.connect(self.B_show_spots_callback)

        ## EMPTY WIDGETS
        for j in range(8, 24):
            _ql = QLabel(win_right)
            L_right.addWidget(_ql, j, 0, alignment=widget_align)

        ## KEYBOARD SHORTCUTS - tab right/left through frames
        self.left_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Left), self.win)
        self.right_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Right), self.win)
        self.left_shortcut.activated.connect(self.tab_prev_frame)
        self.right_shortcut.activated.connect(self.tab_next_frame)

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
        # Update the images
        for i in indices:
            self.LIVs[i].setImage(self.images[i], 
                autoRange=autoRange, autoLevels=autoLevels,
                autoHistogramRange=autoHistogramRange)

        # Update the scatter plots
        if self.B_spot_overlay_state and len(self.detections.shape)==2:
            symbol = self.M_symbol.currentText()
            for j in range(3):
                self.SPIs[j].setData(
                    pos=self.detections+0.5,
                    size=symbol_sizes[symbol], 
                    symbol=symbol, 
                    pen={'color': MASTER_COLOR, 'width': 2},
                    brush=None, pxMode=False
                )
        else:
            for j in range(3):
                self.SPIs[j].setData([])

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
        self.images[0] = self.ChunkFilter.get_subregion(frame_index,
            **self.ChunkFilter.sub_kwargs)

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
            self.detect_sliders[i].configure(minimum=np.percentile(self.images[2], 10), 
                maximum=self.images[2].max())
            self.detect_sliders[i].setValueBlock(self.detect_kwargs['t'])

    def change_roi(self, **roi_kwargs):
        """
        Change the ROI used for filtering.

        """
        # Update the ChunkFilter, which sets ChunkFilter.sub_kwargs
        self.ChunkFilter.set_subregion(**roi_kwargs)
        self.ChunkFilter.load_chunk(self.ChunkFilter.chunk_start)

        # Update the first (raw) image
        self.load_frame(self.frame_slider.value())

        # Run filtering, detection, and update plots
        self.filter()
        self.detect()
        self.update_images(0, 1, 2, 3)

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
        # Get a maximum intensity projection of the image
        if not hasattr(self, "max_int_proj"):
            self.max_int_proj = self.ChunkFilter.max_int_proj(
                start=0, stop=min(1000, self.ChunkFilter.n_frames))

        # Prompt the user to input an ROI
        y_slice, x_slice = PromptSelectROI(self.max_int_proj, parent=self)

        # If the user accepts the selection, proceed
        if not y_slice is None:

            roi_kwargs = {'y0': y_slice.start, 'y1': y_slice.stop-1, 
                'x0': x_slice.start, 'x1': x_slice.stop-1}
            self.change_roi(**roi_kwargs)

    def B_change_frame_range_callback(self):
        """
        Change the frame range.

        """
        # Prompt the user to select a start frame and stop frame
        names = ['Start frame (min 0)', 'Stop frame (max %d)' % \
            (self.ChunkFilter.n_frames-1)]
        defaults = [0, 100]
        start_frame, stop_frame = getTextInputs(names, defaults, title='Select frame range')

        # Update the ChunkFilter
        self.ChunkFilter.start = start_frame
        self.ChunkFilter._init_chunks()
        self.ChunkFilter.load_chunk(start_frame)

        # Update frame slider
        self.frame_slider.configure(minimum=start_frame, maximum=stop_frame,
            interval=1, init_value=start_frame)

        # Rerun filtering and detection        
        self.load_frame(start_frame)
        self.filter()
        self.detect()

        # Update the plots
        self.update_images(0, 1, 2, 3, autoRange=True, autoLevels=True,
            autoHistogramRange=True)

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
        self.auto_threshold()
        self.detect()
        self.update_images(2, 3, autoLevels=True, autoHistogramRange=True)
        if self.B_spot_overlay_state:
            self.update_images(0, 1)

    def B_spot_overlay_callback(self):
        """
        Toggle spot overlay on the image.

        """
        self.B_spot_overlay_state = not self.B_spot_overlay_state 
        self.update_images(0, 1, 2)

    def M_symbol_callback(self):
        """
        Change the spot overlay symbol.

        """
        if self.B_spot_overlay_state:
            self.update_images(0, 1, 2)

    def B_save_callback(self):
        """
        Save the current settings to a config file.

        """
        print("DetectViewer.B_save_callback: not yet implemented")

    def B_show_spots_callback(self):
        """
        Show an array of individual detected spots.

        """
        # The size of the image grid
        N = 10

        # Get as many spots as we can to fill this image
        # grid
        frame_index = self.frame_slider.value()
        positions = []
        images = []
        while sum([i.shape[0] for i in positions])<N**2 and frame_index<self.ChunkFilter.n_frames:
            im = self.ChunkFilter.filter_frame(frame_index)
            pos = detect(im, **self.detect_kwargs)
            images.append(im)
            positions.append(pos)
            frame_index += 1

        # Get as many spots as 
        ex = ImageSubpositionWindow(images, 
            positions, w=15, N=N, parent=self)

    ## KEYBOARD SHORTCUT CALLBACKS

    def tab_next_frame(self):
        """
        Load the next frame.

        """
        next_idx = int(self.frame_slider.value())
        if next_idx < self.frame_slider.maximum - 1:
            next_idx += 1
        self.frame_slider.setValue(next_idx)

    def tab_prev_frame(self):
        """
        Load the previous frame.

        """
        prev_idx = int(self.frame_slider.value())
        if prev_idx > self.frame_slider.minimum:
            prev_idx -= 1
        self.frame_slider.setValue(prev_idx)




