#!/usr/bin/env python
"""
ImageViewer.py -- a simple viewer for image movies

"""
# Image reader
from ..read import ImageReader

# Core GUI utilities
import PySide2
from PySide2.QtCore import Qt 
from PySide2.QtWidgets import QWidget, QGridLayout, \
    QPushButton, QDialog, QLabel, QLineEdit, QShortcut
from PySide2.QtGui import QKeySequence
from PySide2.QtGui import Qt as QtGui_Qt

# pyqtgraph utilities for showing images
from pyqtgraph import ImageView 

# Custom GUI utilities
from .guiUtils import IntSlider, getTextInputs, \
    LabeledQComboBox, coerce_type, SingleImageWindow

class ImageViewer(QWidget):
    """
    Show a single frame from a movie with a slider
    to change the frame. This essentially harnesses 
    pyqtgraph.ImageView for a simple image viewer that is
    occasionally easier than Fiji.

    init
    ----

    """
    def __init__(self, path, parent=None):
        super(ImageViewer, self).__init__(parent=parent)
        self.path = path 
        self.initData()
        self.initUI()

    def initData(self):
        """
        Try to read the image data at the target path.

        """
        self.ImageReader = ImageReader(self.path)

    def initUI(self):
        """
        Initialize the user interface.

        """
        # Main window
        self.win = QWidget()
        self.win.setWindowTitle(self.path)
        layout = QGridLayout(self.win)

        # ImageView
        self.ImageView = ImageView(parent=self.win)
        layout.addWidget(self.ImageView, 0, 0, 2, 2)

        # Frame slider
        self.frame_slider = IntSlider(minimum=0, interval=1,
            maximum=self.ImageReader.n_frames-1, init_value=1,
            name='Frame', parent=self.win)
        layout.addWidget(self.frame_slider, 2, 0, 1, 1, alignment=Qt.AlignTop)
        self.frame_slider.assign_callback(self.frame_slider_callback)

        # Buttons to make projections
        self.B_max_int = QPushButton("Make projection", self.win)
        layout.addWidget(self.B_max_int, 3, 0, 1, 1, alignment=Qt.AlignLeft)
        self.B_max_int.clicked.connect(self.B_max_int_callback)

        # Use the right/left keys to tab through frames
        self.left_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Left), self.win)
        self.right_shortcut = QShortcut(QKeySequence(QtGui_Qt.Key_Right), self.win)
        self.left_shortcut.activated.connect(self.prev_frame)
        self.right_shortcut.activated.connect(self.next_frame)

        # Update the frame
        self.load_frame(0, reset=True)

        # Resize main window
        self.win.resize(600, 600)

        # Show the main window
        self.win.show()

    def load_frame(self, frame_index, reset=False):
        """
        Change the current frame.

        args
        ----
            frame_index     :   int
            reset           :   bool, reset the LUTs and ROI

        """
        self.image = self.ImageReader.get_frame(frame_index)
        self.ImageView.setImage(self.image, autoRange=reset, autoLevels=reset,
            autoHistogramRange=reset)

    def next_frame(self):
        """
        Go to the frame after the current one.

        """
        next_idx = int(self.frame_slider.value())
        if next_idx < self.frame_slider.maximum:
            next_idx += 1
        self.frame_slider.setValue(next_idx)

    def prev_frame(self):
        """
        Go the frame before the current one.

        """
        prev_idx = int(self.frame_slider.value())
        if prev_idx > self.frame_slider.minimum:
            prev_idx -= 1
        self.frame_slider.setValue(prev_idx)

    def frame_slider_callback(self):
        """
        Change the current frame.

        """
        self.load_frame(self.frame_slider.value())

    def B_max_int_callback(self):
        """
        Make a maximum intensity projection.

        """
        ex = ChooseProjectionDialog(self.ImageReader.n_frames, parent=self)
        if ex.exec_() == QDialog.Accepted:
            method, start_frame, stop_frame = ex.return_val

            # Perform the projection
            result = getattr(self.ImageReader, method)(start=int(start_frame),
                stop=int(stop_frame))

            # Make a standalone window showing the projection
            ex = SingleImageWindow(result, title=method, parent=self)
            ex.show()

class ChooseProjectionDialog(QDialog):
    def __init__(self, n_frames, parent=None):
        super(ChooseProjectionDialog, self).__init__(parent=parent)
        self.n_frames = n_frames 
        self.initUI()

    def initUI(self):
        layout = QGridLayout(self)
        self.setWindowTitle("Select projection")

        # Menu to select type of projection
        proj_types = ['max_int_proj', 'sum_proj']
        self.M_proj = LabeledQComboBox(proj_types, "Projection type",
            init_value="max_int_proj", parent=self)
        layout.addWidget(self.M_proj, 0, 0, 1, 2, alignment=Qt.AlignRight)

        # Entry boxes to choose start and stop frames
        label_0 = QLabel(self)
        label_1 = QLabel(self)
        label_0.setText("Start frame")
        label_1.setText("Stop frame")
        layout.addWidget(label_0, 1, 0, alignment=Qt.AlignRight)
        layout.addWidget(label_1, 2, 0, alignment=Qt.AlignRight)

        self.EB_0 = QLineEdit(self)
        self.EB_1 = QLineEdit(self)
        self.EB_0.setText(str(0))
        self.EB_1.setText(str(self.n_frames))
        layout.addWidget(self.EB_0, 1, 1, alignment=Qt.AlignLeft)
        layout.addWidget(self.EB_1, 2, 1, alignment=Qt.AlignLeft)

        # Accept button
        self.B_accept = QPushButton("Accept", parent=self)
        self.B_accept.clicked.connect(self.B_accept_callback)
        layout.addWidget(self.B_accept, 3, 0, alignment=Qt.AlignRight)

    def B_accept_callback(self):
        """
        Accept the current projection settings and return to the 
        client widget.

        """
        try:
            self.return_val = [
                self.M_proj.currentText(),
                coerce_type(self.EB_0.text(), int),
                coerce_type(self.EB_1.text(), int),
            ]
            self.accept()
        except ValueError:
            print("Frame values must be integers")








