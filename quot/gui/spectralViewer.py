#!/usr/bin/env python
"""
spectralViewer.py -- visualize different parts of an image's
power spectrum

"""
import os
import numpy as np 

from ..read import ImageReader 

# Core GUI utilities
import PySide2
from PySide2.QtCore import Qt 
from PySide2.QtWidgets import QApplication, QWidget, QLabel, \
    QPushButton, QGridLayout, QVBoxLayout, QDialog, QShortcut
from PySide2.QtGui import Qt as QtGui_Qt
from PySide2.QtGui import QKeySequence

from pyqtgraph import ImageView 

from .guiUtils import (
    FloatSlider,
    IntSlider,
    LabeledImageView,
    LabeledQComboBox
)

class SpectralViewer(QWidget):
    def __init__(self, path, start_frame=0, gui_size=900, 
        pixel_size_um=0.16, parent=None):
        self.path = path 
        self.start_frame = start_frame 
        self.gui_size = gui_size 
        self.pixel_size_um = pixel_size_um 

        self.initData()

        self.initUI()

    def initData(self):
        self.reader = ImageReader(self.path)

        # Mean projection
        if self.reader.n_frames > 0:
            self.mean_proj = self.reader.sum_proj(start_frame=self.start_frame) / (self.reader.n_frames - self.start_frame)
        else:
            self.mean_proj = self.reader.get_frame(0).astype(np.float64)

        # Kill the DC component
        self.mean_proj -= self.mean_proj.mean()

        Ny, Nx = self.mean_proj.shape 
        self.Ny = Ny 
        self.Nx = Nx 

        # Frequencies
        self.yfreq = np.fft.fftfreq(Ny, d=self.pixel_size_um)
        self.xfreq = np.fft.fftfreq(Nx, d=self.pixel_size_um)
        self.freq_interval = np.abs((self.yfreq[1:] - self.yfreq[:-1])).min()
        Xfreq, Yfreq = np.meshgrid(self.xfreq, self.yfreq)
        self.rfreq = np.sqrt(Xfreq**2 + Yfreq**2)
        self.xfreq = Xfreq 
        self.yfreq = Yfreq 
        self.max_freq = int(np.ceil(self.rfreq.max()))

        # Filter
        self.take_freqs = np.ones(self.rfreq.shape, dtype=np.bool)

        # The actual substrate for showing / filtering
        self.I = self.mean_proj.copy()
        self.I_ft = np.fft.fft2(self.I)
        self.I_ft_filt = self.I_ft.copy()

    def initUI(self):

        # Master layout
        self.win = QWidget()
        L_master = QGridLayout(self.win)

        # Increase the Qt column stretch for the first column, which 
        # encodes the actual images
        L_master.setColumnStretch(0, 1)

        # Assign the close event
        self.win.closeEvent = self.closeEvent 

        # Left subwindow, for widgets
        self.win_left = QWidget(self.win)
        L_left = QGridLayout(self.win_left)
        L_master.addWidget(self.win_left, 0, 0, 1, 1)

        # Right subwindow, for images
        self.win_right = QWidget(self.win)
        L_right = QGridLayout(self.win_right)
        L_master.addWidget(self.win_right, 0, 1, 1, 1)

        # Two ImageView objects to contain the original and
        # displayed images
        titles = [
            "Original image",
            "Selected band"
        ]
        self.labeledImageViews = [
            LabeledImageView(
                parent=self.win_left,
                label=titles[j]
            ) for j in range(2)
        ]
        self.imageViews = [self.labeledImageViews[j].ImageView for j in range(2)]

        # Assign to the left window
        for j in range(2):
            L_left.addWidget(self.labeledImageViews[j], j%2, j//2)

        # Link every movement in X and Y between the different ImageView
        # objects
        for j in range(1, 2):
            self.imageViews[j].view.setXLink(self.imageViews[0].view)
            self.imageViews[j].view.setYLink(self.imageViews[0].view)

        # Widgets
        widget_align = Qt.AlignAbsolute
        slider_min_width = 125

        # Frequency band definition
        self.band_0_lower_slider = FloatSlider(
            parent=self.win_right,
            name="Lower",
            minimum=0.0,
            maximum=self.max_freq,
            interval=self.freq_interval,
            init_value=0.0,
            min_width=slider_min_width,
        )
        self.band_0_upper_slider = FloatSlider(
            parent=self.win_right,
            name="Upper",
            minimum=0.0,
            maximum=self.max_freq,
            interval=self.freq_interval,
            init_value=self.max_freq,
            min_width=slider_min_width,
        )
        self.band_0_lower_slider.assign_callback(self.adjust_band)
        self.band_0_upper_slider.assign_callback(self.adjust_band)
        L_right.addWidget(self.band_0_lower_slider, 2, 0)
        L_right.addWidget(self.band_0_upper_slider, 3, 0)

        # Button to select either mean projection / individual frames
        self.combo_box_mode = LabeledQComboBox(
            ["Mean projection", "Individual frames"],
            "Mode",
            init_value="Mean projection",
            parent=self.win_right,
        )
        self.combo_box_mode.assign_callback(self.combo_box_mode_callback)
        L_right.addWidget(self.combo_box_mode, 0, 0)

        # Button to auto-range the LUTs
        self.B_auto_range = QPushButton("LUT auto range", parent=self.win_right)
        self.B_auto_range.clicked.connect(self.auto_range_callback)
        L_right.addWidget(self.B_auto_range, 1, 0)

        # Frame slider
        self.frame_slider = IntSlider(
            minimum=0, 
            maximum=self.reader.n_frames-1, 
            interval=1, 
            name="Frame",
            init_value=self.start_frame,
            min_width=slider_min_width,
            parent=self.win_right 
        )
        self.frame_slider.assign_callback(self.frame_slider_callback)
        L_right.addWidget(self.frame_slider, 4, 0)
        self.frame_slider.hide()

        self.imageViews[0].setImage(self.I)
        self.adjust_band(autoThresh=True)

        self.win.show()

    def adjust_band(self, autoThresh=False):
        """
        Change the frequency band being displayed.

        """
        t0 = self.band_0_lower_slider.value()
        t1 = self.band_0_upper_slider.value()

        # Ensure that t0 < t1
        if t1 < t0:
            t2 = t1 
            t1 = t0 
            t0 = t2
        elif t1 == t0:
            t0 = t0 - self.freq_interval * 0.5
            t1 = t1 + self.freq_interval * 0.5

        self.filter_image(t0, t1)

        self.imageViews[1].setImage(
            self.I_conv,
            autoRange=autoThresh,
            autoLevels=autoThresh,
            autoHistogramRange=autoThresh
        )

    def filter_image(self, t0, t1):
        """
        Take the band between frequencies *t0* and *t1*.

        """
        self.take_freqs = np.logical_and(
            self.rfreq >= t0, 
            self.rfreq < t1 
        )
        self.I_ft_filt[:,:] = 0
        self.I_ft_filt[self.take_freqs] = self.I_ft[self.take_freqs]
        self.I_conv = np.real(np.fft.ifft2(self.I_ft_filt, s=self.I.shape))

    def update_primary_image(self, autoThresh=True):
        self.imageViews[0].setImage(
            self.I,
            autoRange=autoThresh,
            autoLevels=autoThresh,
            autoHistogramRange=autoThresh
        )

    def combo_box_mode_callback(self):

        mode = self.combo_box_mode.currentText()

        if mode == "Mean projection":
            self.I = self.reader.sum_proj(start_frame=self.start_frame) / (self.reader.n_frames - self.start_frame)
            self.frame_slider.hide()

        elif mode == "Individual frames":
            self.I = self.reader.get_frame(int(self.frame_slider.value()))
            self.frame_slider.show()

        self.I -= self.I.mean()
        self.I_ft = np.fft.fft2(self.I)
        self.update_primary_image()
        self.adjust_band()

    def frame_slider_callback(self):
        if self.combo_box_mode.currentText() == "Individual frames":
            frame_index = int(self.frame_slider.value())
            self.I = self.reader.get_frame(frame_index)
            self.I -= self.I.mean()
            self.I_ft = np.fft.fft2(self.I)
            self.update_primary_image()
            self.adjust_band()

    def auto_range_callback(self):
        self.imageViews[0].setImage(
            self.I, 
            autoRange=True,
            autoLevels=True,
            autoHistogramRange=True 
        )
        self.imageViews[1].setImage(
            self.I_conv, 
            autoRange=True,
            autoLevels=True,
            autoHistogramRange=True 
        )

    def closeEvent(self, event):
        """
        Close this window and the file reader.

        """
        self.reader.close()
        self.win.close()
        # self.close()
        event.accept()

