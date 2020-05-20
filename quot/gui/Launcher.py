#!/usr/bin/env python
"""
Launcher.py -- a simple menu that provides options to launch
other, more specialized GUIs

"""
import sys

# File paths
import os 

# Main GUI utilities
from PySide2.QtCore import Qt 
from PySide2.QtWidgets import QApplication, QWidget, QLabel, \
    QPushButton, QVBoxLayout

# Custom GUI utilities
from .guiUtils import set_dark_app, getOpenFilePath
from .ImageViewer import ImageViewer 
from .DetectViewer import DetectViewer 

class Launcher(QWidget):
    """
    Simple GUI that presents the user with a list of options
    to launch other GUIs. Pressing on a button will prompt
    the user to select files or settings required to launch
    the corresponding GUI.

    init
    ----
        parent      :   root QWidget, if any 
        currdir     :   str, the default directory for file
                        selection dialogs

    """
    def __init__(self, currdir=None, parent=None):
        super(Launcher, self).__init__(parent)

        # Set default directory for file selection dialogs
        if currdir is None:
            currdir = os.getcwd()
        self.currdir = currdir

        self.initUI()

    def initUI(self):
        """
        Initialize the user interface.

        """
        # Main vertical box layout
        L = QVBoxLayout(self)

        # Title
        title = QLabel(parent=self)
        title.setText("Single particle tracking utilities")
        L.addWidget(title, 0, alignment=Qt.AlignTop)

        # Buttons to launch subGUIs
        button_ids = [
            ("Simple image viewer", self.launch_image_viewer),
            ("Detection viewer", self.launch_detect_viewer),
            ("Spot viewer", self.launch_spot_viewer),
            ("Track viewer", self.launch_track_viewer),
            ("Batch localize", self.launch_batch_localizer),
            ("Attribute viewer", self.launch_attribute_viewer),
        ]
        self.buttons = []
        for i, (label, callback) in enumerate(button_ids):
            self.buttons.append(QPushButton(label, parent=self))
            L.addWidget(self.buttons[-1], i+1, alignment=Qt.AlignTop)
            self.buttons[-1].clicked.connect(callback)

        # Show the window
        self.show()

    def launch_image_viewer(self):
        """
        Launch an instance of ImageViewer on a sample file, 
        which is a simple viewer for movies.

        """
        # Prompt the user to enter a target image file
        path = getOpenFilePath(self, "Select image file",
            "Image files (*.nd2 *.tif *.tiff)",
            initialdir=self.currdir)

        # If this is a real file, launch an ImageViewer on it
        if os.path.isfile(path):
            I = ImageViewer(path, parent=self)

    def launch_detect_viewer(self):
        """
        Launch an instance of DetectViewer on a sample file.

        """
        # Prompt the user to enter a target image file
        path = getOpenFilePath(self, "Select image file",
            "Image files (*.nd2 *.tif *.tiff)",
            initialdir=self.currdir)

        # If this is a real file path, launch a DetectViewer 
        # on it
        if os.path.isfile(path):
            V = DetectViewer(path, parent=self)


    def launch_spot_viewer(self):
        pass 

    def launch_track_viewer(self):
        pass 

    def launch_batch_localizer(self):
        pass 

    def launch_attribute_viewer(self):
        pass 

def init_launcher():
    """
    Initialize a standalone instance of Launcher.

    """
    app = QApplication()
    set_dark_app(app)
    ex = Launcher()
    sys.exit(app.exec_())

if __name__ == '__main__':
    init_launcher()
