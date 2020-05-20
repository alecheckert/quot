#!/usr/bin/env python
"""
guiUtils.py -- helper utilities for the quot GUI

"""
import sys 

# File paths
import os

# Numeric
import numpy as np 

# Main GUI utilities
from PySide2.QtCore import Qt, QSize, QRectF
from PySide2.QtGui import QPalette, QColor, QPainterPath
from PySide2.QtWidgets import QFileDialog, QSlider, QWidget, \
    QGridLayout, QVBoxLayout, QLabel, QPushButton, QLineEdit, \
    QDialog, QComboBox 

# pyqtgraph utilities
from pyqtgraph import ImageView, GraphicsLayoutWidget, RectROI, \
    ImageItem 
from pyqtgraph.pgcollections import OrderedDict 

# Master color for all ROIs, spot overlays, etc.
# Potentially good: #88BDF6, #00DBAB
MASTER_COLOR = '#88BDF6'

###############
## FUNCTIONS ##
###############

def set_dark_app(qApp):
    """
    Set the color scheme of a PySide2 QApplication to 
    a darker default, inherited by all children. 

    Modifies the QApplication in place.

    """
    qApp.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    qApp.setPalette(palette)
    qApp.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")

def keys_to_str(dict_keys):
    """
    Return the set of keys to a Python dictionary
    as a list of str.

    args
    ----
        dict_keys   :   class dict_keys

    returns
    -------
        list of str

    """
    return [str(j) for j in dict_keys]

def coerce_type(arg, type_):
    """
    Try to coerce a string into the type class *type_*.
    When this fails, a ValueError will be raised.

    args
    ----
        arg     :   str
        type_   :   a type class, like int or float

    returns
    -------
        type_, if arg can be coerced, or *arg* if 
            coercion fails

    """
    if type_ is int:
        return int(arg)
    elif type_ is float:
        return float(arg)
    elif type_ is bool:
        return bool(arg)
    elif type_ is str:
        return str(arg)

#############
## WIDGETS ##
#############

class LabeledImageView(QWidget):
    """
    A pyqtgraph.ImageView instance with a title above it,
    to distinguish it in situations with multiple ImageViews.

    init
    ----
        parent          :   root QWidget
        label           :   str
        **kwargs        :   to ImageView.__init__

    """
    def __init__(self, label=None, parent=None, **kwargs):
        super(LabeledImageView, self).__init__(parent)
        self.label_text = label 
        self.initUI(**kwargs)

    def initUI(self, **kwargs):
        """
        Initialize the user interface.

        """
        self.layout = QVBoxLayout(self)
        self.title_widget = QLabel(self)
        self.title_widget.setText(self.label_text)
        self.ImageView = ImageView(**kwargs)
        self.layout.addWidget(self.title_widget, 0)
        self.layout.addWidget(self.ImageView, 0)

    def setImage(self, *args, **kwargs):
        """
        Convenience wrapper for ImageView.setImage.

        """
        self.ImageView.setImage(*args, **kwargs)

    def sizeHint(self):
        """
        Recommended size of this QWidget to the 
        Qt overlords.

        """
        return QSize(300, 300)

class LabeledQComboBox(QWidget):
    """
    A QComboBox with a QLabel above it. Useful to indicate
    the title of a variable represented by this QComboBox.

    init
    ----
        parent          :   root QWidget
        options         :   list of str, values for the QComboBox
        label           :   str, the title above the box
        init_value      :   str, starting value

    """
    def __init__(self, options, label, init_value=None, parent=None):
        super(LabeledQComboBox, self).__init__(parent)
        self.options = options 
        self.label_text = label
        self.init_value = init_value
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.QComboBox = QComboBox(parent=self)
        self.QComboBox.addItems(self.options)
        self.label_widget = QLabel(parent=self)
        self.label_widget.setText(self.label_text)
        self.layout.addWidget(self.label_widget, 0, alignment=Qt.AlignTop)
        self.layout.addWidget(self.QComboBox, 1, alignment=Qt.AlignTop)

        if not self.init_value is None:
            self.QComboBox.setCurrentText(self.init_value)

    def sizeHint(self):
        """
        Recommended size of this QWidget to the 
        Qt overlords.

        """
        return QSize(200, 100)

    def currentText(self):
        return self.QComboBox.currentText()

    def setCurrentText(self, *args, **kwargs):
        self.QComboBox.setCurrentText(*args, **kwargs)

    def setLabel(self, text):
        self.label_widget.setText(str(text))

    def assign_callback(self, func):
        self.QComboBox.activated.connect(func)

class IntSlider(QWidget):
    """
    An integer-valued slider widget with increment 1, to
    be added to a larger GUI. Included is (a) a label for
    the minimum value the slider can assume, (b) a label
    for the maximum value the slider can assume, (c) the 
    current slider value, and (d) a name for the slider.

    init
    ----
        parent          :   root QWidget
        minimum         :   int, minimum value for the slider
        maximum         :   int, maximum value for the slider
        interval        :   int, the interval between ticks
        init_value      :   int, initial value
        name            :   str

    """
    def __init__(self, minimum=0, maximum=10, interval=1, 
        init_value=0, name=None, min_width=150, parent=None):
        super(IntSlider, self).__init__(parent=parent)

        self.minimum = int(minimum)
        self.maximum = int(maximum)
        self.interval = int(interval)
        self.min_width = min_width 
        if name is None:
            name = ''
        self.name = name 

        # If the interval is not 1, figure out whether 
        # the maximum needs to be decreased for an integral
        # number of intervals
        self.slider_values = self._set_slider_values(self.minimum,
            self.maximum, self.interval)
        self.maximum = self.slider_values.max()

        self.initUI()

    def _set_slider_values(self, minimum, maximum, interval):
        """
        Configure the values of the slider, useful when the
        interval is not unity.

        """
        if interval != 1:
            n_intervals = (maximum-minimum)//interval + 1
            slider_values = minimum + interval * \
                np.arange(n_intervals).astype(np.int64)
            slider_values = slider_values[slider_values<=maximum]
        else:
            slider_values = np.arange(minimum, maximum+1).astype(np.int64)

        return slider_values

    def initUI(self):
        """
        Initialize the user interface.

        """
        # Main window
        self.win = QWidget(self)
        self.layout = QGridLayout()
        self.win.setLayout(self.layout)

        # Current value / title label
        self.L_title = QLabel(self.win)
        self.layout.addWidget(self.L_title, 0, 1)

        # Label for minimum and maximum
        self.L_min = QLabel(self.win)
        self.L_max = QLabel(self.win)
        self.L_min.setText(str(self.minimum))
        self.L_max.setText(str(self.maximum))
        self.layout.addWidget(self.L_min, 1, 0, alignment=Qt.AlignLeft)
        self.layout.addWidget(self.L_max, 1, 2, alignment=Qt.AlignRight)

        # Base QSlider
        self.slider = QSlider(Qt.Horizontal, self.win)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.slider_values)-1)
        self.layout.addWidget(self.slider, 1, 1)
        self.slider.valueChanged.connect(self._set_label_current)

        # Set a minimum width for this slider
        self.slider.setMinimumWidth(self.min_width)

        # Update the current label 
        self._set_label_current()

    def sizeHint(self):
        """
        Recommended size of this QWidget to the 
        Qt overlords.

        """
        return QSize(200, 100)

    def _set_label_current(self):
        """
        Update the slider title with the current value of 
        self.slider.

        """
        self.L_title.setText("%s: %d" % (self.name, self.value()))

    def value(self):
        """
        Return the current value of the slider as an integer.

        """
        return self.slider_values[self.slider.value()]

    def assign_callback(self, func):
        """
        Trigger a function to be called when the slider is changed.
        If several functions are assigned by sequential uses of this
        method, then all of the functions are executed when the slider
        is changed.

        args
        ----
            func        :   function, no arguments

        """
        self.slider.valueChanged.connect(func)

    def hide(self):
        """
        Hide this IntSlider.

        """
        self.win.hide()

    def show(self):
        """
        Show this IntSlider.

        """
        self.win.show()

    def isVisible(self):
        """
        Returns False if the IntSlider is currently hidden.

        """
        return self.win.isVisible()

    def toggle_vis(self):
        """
        Toggle the visibility of this IntSlider between hidden
        and shown.

        """
        if self.isVisible():
            self.hide()
        else:
            self.show()

    def configure(self, **kwargs):
        """
        Change some or all of the slider attributes. Accepted
        kwargs are:

            minimum     :   int, the minimum value of the slider
            maximum     :   int, the maximum value of the slider
            interval    :   int, the slider interval
            name        :   str, the slider label
            init_value  :   int, the initial value

        """
        keys = kwargs.keys()

        # Reconfigure the slider values
        if 'minimum' in keys:
            minimum = int(kwargs.get('minimum'))
        else:
            minimum = self.minimum 
        if 'maximum' in keys:
            maximum = int(kwargs.get('maximum'))
        else:
            maximum = self.maximum 
        if 'interval' in keys:
            interval = int(kwargs.get('interval'))
        else:
            interval = self.interval 

        self.slider_values = self._set_slider_values(minimum,
            maximum, interval)
        self.maximum = self.slider_values.max()
        self.minimum = minimum 
        self.interval = interval 

        # Update the QSlider
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.slider_values)-1)

        # Set name
        if 'name' in keys:
            self.name = kwargs.get('name')

        # Set initial value
        if 'init_value' in keys:
            midx = (np.argmin(np.abs(self.slider_values - \
                kwargs.get('init_value'))))
            self.slider.setValue(midx)

        # Update labels
        self.L_min.setText(str(self.minimum))
        self.L_max.setText(str(self.maximum))
        self._set_label_current()

class FloatSlider(QWidget):
    """
    A float-valued slider widget to be added to a larger
    GUI.

    Includes:
        - a label for the minimum value the slider can assume
        - a label for the maximum value the slider can assume
        - a label for the current slider value
        - a name for the slider

    init
    ----
        parent          :   root QWidget
        minimum         :   int, minimum value for the slider
        maximum         :   int, maximum value for the slider
        interval        :   int, the interval between ticks
        init_value      :   int, initial value
        name            :   str
        min_width       :   int, the minimum width for this widget
        return_int      :   bool, return values are integers
                            rather than floats (sometimes useful)

    """
    def __init__(self, minimum=0.0, maximum=10.0, interval=1.0, 
        init_value=0, name=None, min_width=150, parent=None,
        return_int=False):
        super(FloatSlider, self).__init__(parent)

        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.interval = float(interval)
        self.init_value = float(init_value)
        self.min_width = min_width 
        self.return_int = return_int

        # Choose the floating point precision 
        if (not '.' in str(interval)) or return_int:
            self.precision = 0
        else:
            self.precision = len(str(interval).split('.')[1])

        if name is None:
            name = ''
        self.name = name 

        # Configure the slider values and possibly adjust
        # the max so it is an integral multiple of the interval
        # plus minimum
        self.slider_values = self._get_slider_values(self.minimum,
            self.maximum, self.interval)
        self.maximum = self.slider_values.max()

        self.initUI()

    def initUI(self):
        """
        Initialize the user interface.

        """
        # Main window
        self.win = QWidget(self)
        self.layout = QGridLayout(self.win)

        # Current value / title label
        self.L_title = QLabel(self)
        self.layout.addWidget(self.L_title, 0, 1)

        # Label for minimum and maximum
        self.L_min = QLabel(self)
        self.L_max = QLabel(self)
        self.L_min.setText("%.1f" % self.minimum)
        self.L_max.setText("%.1f" % self.maximum)
        self.layout.addWidget(self.L_min, 1, 0, alignment=Qt.AlignLeft)
        self.layout.addWidget(self.L_max, 1, 2, alignment=Qt.AlignRight)

        # Base QSlider
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.slider_values)-1)
        self.layout.addWidget(self.slider, 1, 1)
        self.slider.valueChanged.connect(self._set_label_current)

        # Set a minimum width for this slider
        self.slider.setMinimumWidth(self.min_width)

        # Set the initial slider value
        self.setValue(self.init_value)

        # Update the current label 
        self._set_label_current()

    def sizeHint(self):
        """
        Recommended size of this QWidget to the 
        Qt overlords.

        """
        return QSize(200, 100)

    def _set_label_current(self):
        """
        Update the current label text.

        """
        self.L_title.setText("%s: %s" % (self.name,
            ("{0:.%df}" % self.precision).format(self.value())))

    def _get_slider_values(self, minimum, maximum, interval):
        """
        Get the discrete values this slider can assume.

        args
        ----
            minimum, maximum, interval  :   float

        returns
        -------
            1D ndarray, the slider values

        """
        n_intervals = (maximum-minimum)//interval + 1
        slider_values = minimum + interval*np.arange(n_intervals)
        slider_values = slider_values[slider_values<=maximum]
        return slider_values 

    def value(self):
        """
        Get the slider's current value.

        """
        if self.return_int:
            return int(self.slider_values[int(self.slider.value())])
        else:
            return self.slider_values[int(self.slider.value())]

    def setValue(self, value):
        """
        Set the slider to the closest available value.

        """
        m = np.argmin(np.abs(float(value)-self.slider_values))
        self.slider.setValue(m)

    def setValueBlock(self, value):
        """
        Set the slider to the closest available value, 
        suppressing calls to linked functions.

        """
        self.slider.blockSignals(True)
        self.setValue(value)
        self.slider.blockSignals(False)
        self._set_label_current()

    def assign_callback(self, func):
        """
        Trigger a function to be called when the slider is changed.
        If several functions are assigned by sequential uses of this
        method, then all of the functions are executed when the slider
        is changed.

        args
        ----
            func        :   function, no arguments

        """
        self.slider.valueChanged.connect(func)

    def hide(self):
        """
        Hide this FloatSlider.

        """
        self.win.hide()

    def show(self):
        """
        Show this FloatSlider.

        """
        self.win.show()

    def isVisible(self):
        """
        Returns False if the FloatSlider is currently hidden.

        """
        return self.slider.isVisible()

    def toggle_vis(self):
        """
        Toggle the visibility of this FloatSlider between hidden
        and shown.

        """
        if self.isVisible():
            self.hide()
        else:
            self.show()

    def configure(self, **kwargs):
        """
        Change some or all of the slider attributes. Accepted
        kwargs are:

            minimum     :   float, the minimum value of the slider
            maximum     :   float, the maximum value of the slider
            interval    :   float, the slider interval
            name        :   str, the slider label
            init_value  :   float, the initial value
            return_int  :   bool, whether to return the result as an 
                            integer

        """
        keys = kwargs.keys()

        # Reconfigure the slider values
        if 'minimum' in keys:
            minimum = float(kwargs.get('minimum'))
        else:
            minimum = self.minimum 
        if 'maximum' in keys:
            maximum = float(kwargs.get('maximum'))
        else:
            maximum = self.maximum 
        if 'interval' in keys:
            interval = float(kwargs.get('interval'))
        else:
            interval = self.interval 

        self.slider_values = self._get_slider_values(
            minimum, maximum, interval)
        self.maximum = self.slider_values.max()
        self.minimum = minimum 
        self.interval = interval 

        # Update the QSlider
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.slider_values)-1)

        # Set name
        if 'name' in keys:
            self.name = kwargs.get('name')

        # Set initial value
        if 'init_value' in keys:
            self.setValue(kwargs.get('init_value'))

        # Return as int if desired
        self.return_int = kwargs.get('return_int', self.return_int)

        # Set precision for float representations
        if self.return_int or (not '.' in str(self.interval)):
            self.precision = 0
        else:
            self.precision = len(str(self.interval).split('.')[1])

        # Update labels
        self.L_min.setText("%.1f" % self.minimum)
        self.L_max.setText("%.1f" % self.maximum)
        self._set_label_current()

class ROISelectionBox(QDialog):
    """
    An ImageItem with a rectangular ROI that allows 
    selection of a subregion.

    """
    def __init__(self, image, parent=None):
        super(ROISelectionBox, self).__init__(parent=parent)
        self.image = image 
        self.initUI()

    def initUI(self):

        # Master window
        lay = QGridLayout(self)
        self.setWindowTitle("Select ROI")

        # Graphics layout, for showing image and ROI
        self.w = GraphicsLayoutWidget(show=True, size=(800, 800), 
            border=True, parent=self)
        lay.addWidget(self.w, 0, 0)
        w1 = self.w.addLayout(row=0, col=0)
        vb = w1.addViewBox(row=0, col=0, lockAspect=True)
        self.II = ImageItem(self.image)
        vb.addItem(self.II)
        self.roi = RectROI([0, 0], [self.image.shape[0], self.image.shape[1]],
            pen={'color': MASTER_COLOR, 'width': 2})
        vb.addItem(self.roi)

        # Button to accept current ROI
        self.B_accept = QPushButton("Accept", parent=self)
        lay.addWidget(self.B_accept, 1, 0, alignment=Qt.AlignLeft)
        self.B_accept.clicked.connect(self.B_accept_callback)

        # Show the main window
        self.resize(self.image.shape[0], self.image.shape[1])
        self.show()

    def B_accept_callback(self):
        self.return_val = self.roi.getArraySlice(self.image, self.II)[0]
        self.accept()

def PromptSelectROI(image, parent=None):
    """
    Convenience function. Use ROISelectBox to prompt the user
    to select an ROI from an image.

    args
    ----
        image       :   2D ndarray (YX)
        parent      :   root QWidget

    returns
    -------
        (slice y, slice x), the rectangular ROI slice

    """
    ex = ROISelectionBox(image, parent=parent)
    if ex.exec_() == QDialog.Accepted:
        y_slice, x_slice = ex.return_val
        return y_slice, x_slice
    else:
        return None, None

###################################
## DIALOGS - FILE SELECTION ETC. ##
###################################

class TextEntryDialog(QDialog):
    """
    Prompt the user to modify entry boxes with data. 

    The user sees several labels on the left, each
    corresponding to an entry box on the right. The 
    entry box contains some default text that the 
    user can modify prior to selecting "Accept".

    init
    ----
        item_labels     :   list of str, labels to show
                            on the left
        defaults        :   list of values (variable type),
                            initial values for each entry
                            box. Also determines the expected
                            return type for each argument
        title           :   str, title for this window dialog
        parent          :   root QWidget

    """
    def __init__(self, item_labels, defaults, title=None, parent=None):
        super(TextEntryDialog, self).__init__(parent)
        self.item_labels = item_labels
        self.defaults = defaults 
        self.title = title 
        self.n_labels = len(self.defaults)
        assert self.n_labels == len(self.item_labels)
        self.return_types = [type(i) for i in self.defaults]

        self.initUI()

    def initUI(self):
        if not (self.title is None):
            self.setWindowTitle(self.title)

        layout = QGridLayout(self)

        # Labels on the left 
        for i, label in enumerate(self.item_labels):
            lw = QLabel(self)
            lw.setText(label)
            layout.addWidget(lw, i, 0, alignment=Qt.AlignRight)

        # Entry boxes on the right
        self.entries = []
        for i in range(self.n_labels):
            self.entries.append(QLineEdit(parent=self))
            layout.addWidget(self.entries[-1], i, 1,
                alignment=Qt.AlignLeft)
            if not self.defaults is None:
                self.entries[-1].setText(str(self.defaults[i]))

        # Button: accept the current entries
        self.B_accept = QPushButton("Accept", parent=self)
        layout.addWidget(self.B_accept, self.n_labels, 0)
        self.B_accept.clicked.connect(self.B_accept_callback)

    def B_accept_callback(self):

        # Check whether the types are correct
        values = [i.text() for i in self.entries]
        try:
            self.return_val = [
                coerce_type(value, self.return_types[i]) \
                    for i, value in enumerate(values)
                ]
            self.accept()
        except ValueError:
            print("Type error; cannot accept these types")

def getTextInputs(item_labels, defaults, title=None):
    """
    Launch an instance of TextEntryDialog to get some input
    from the user, and return the result. Intended to be 
    launched from another GUI.

    args
    ----
        item_labels     :   list of str, names of each parameter
                            to display to the user
        defaults        :   list (variable type), the initial
                            values of each variable to show, which
                            also determines their expected return type
        title           :   str, title for the window

    returns
    -------
        list (variable type), the user input

    """
    ex = TextEntryDialog(item_labels, defaults, title=title)
    if ex.exec_() == QDialog.Accepted:
        return ex.return_val
    else:
        return defaults 

def getOpenFilePath(parent, title, filetypes, initialdir=''):
    """
    Wrapper for PySide2.QtWidgets.QFileDialog.getOpenFileName.
    Prompt the user to select a single file of a particular
    extension.

    args
    ----
        parent          :   root QWidget
        title           :   str, title for dialog window
        filetypes       :   str, e.g. "Image file (*.tif *.tiff)"
        initialdir      :   str, start directory for dialog

    returns
    -------
        str, path

    """
    q = QWidget(parent)
    dialog_options = QFileDialog.Options()
    dialog_options |= QFileDialog.DontUseNativeDialog
    filepath, filetype = QFileDialog.getOpenFileName(
        parent=q, caption=title, filter=filetypes, 
        dir=initialdir, options=dialog_options)
    return filepath

def getOpenFilePaths(parent, title, filetypes, initialdir=''):
    """
    Wrapper for PySide2.QtWidgets.QFileDialog.getOpenFileNames.
    Prompt the user to select one or more files of a particular
    extension.

    args
    ----
        parent          :   root QWidget
        title           :   str, title for dialog window
        filetypes       :   str, e.g. "Image files (*.tif *.tiff)"
        initialdir      :   str, start directory for dialog

    returns
    -------
        list of str, paths

    """
    q = QWidget(parent)
    dialog_options = QFileDialog.Options()
    dialog_options |= QFileDialog.DontUseNativeDialog
    filepaths, filetype = QFileDialog.getOpenFileNames(
        parent=q, caption=title, filter=filetypes, 
        dir=initialdir, options=dialog_options)
    return filepaths

def getSaveFilePath(parent, title, default, filetypes, 
    initialdir=''):
    """
    Wrapper for PySide2.QtWidgets.QFileDialog.getSaveFileName.
    Prompt the user to select a file that doesn't exist to save
    to.

    args
    ----
        parent              :   root QWidget
        title               :   str, title for dialog window
        default             :   str, placeholder text
        filetypes           :   str, accepted file types e.g.
                                "Image files (*.tif *.tiff *.nd2)"
        initialdir          :   str, start directory for dialog

    Returns
    -------
        str, path

    """
    q = QWidget(parent)
    dialog_options = QFileDialog.Options()
    dialog_options |= QFileDialog.DontUseNativeDialog
    path, filetype = QFileDialog.getSaveFileName(
        parent=q, caption=title, default=default,
        filter=filetypes, dir=initialdir, options=dialog_options)
    return path 

def getOpenDirectory(parent, title, initialdir=''):
    """
    Wrapper for PySide2.QtWidgets.QFileDialog.getExistingDirectory.
    Prompt the user to select a directory and return its 
    path as a string.

    args
    ----
        parent          :   root QWidget
        title           :   str, title for dialog window
        initialdir      :   str, start directory for dialog

    returns
    -------
        str, path

    """
    q = QWidget(parent)
    dialog_options = QFileDialog.Options()
    dialog_options |= QFileDialog.DontUseNativeDialog
    result = QFileDialog.getExistingDirectory(
        parent=q, caption=title, dir=initialdir,
        options=dialog_options)
    return result 


#########################
## CUSTOM SPOT SYMBOLS ##
#########################

# These can be used as values for the pyqtgraph.ScatterPlotItem
# "symbol" argument
Symbols = OrderedDict([(s, QPainterPath()) for s in \
    ['o', 's', '+', 'alt +', 'open +']])

# A simple circle around the point
Symbols['o'].addEllipse(QRectF(-0.5, -0.5, 1, 1))

# Square box around the point
Symbols['s'].addRect(QRectF(-0.5, -0.5, 1, 1))

# Crosshairs
coords = {
    '+': [(-0.5, -0.05), (-0.5, 0.05), (-0.05, 0.05), (-0.05, 0.5),
        (0.05, 0.5), (0.05, 0.05), (0.5, 0.05), (0.5, -0.05),
        (0.05, -0.05), (0.05, -0.5), (-0.05, -0.5), (-0.05, -0.05)],
    'alt +': [(-0.5, 0.0), (0.5, 0.0), (0.0, 0.0), (0.0, 0.5),
        (0.0, -0.5)],
}
for k, c in coords.items():
    Symbols[k].moveTo(*c[0])
    for x, y in c[1:]:
        Symbols[k].lineTo(x, y)

# Open crosshairs (empty in the middle)
for from_, to_ in [[(-0.5, 0.0), (-0.25, 0)], [(0.5,0.0), (0.25,0)],
    [(0.0,-0.5), (0.0,-0.25)], [(0.0,0.5), (0.0,0.25)]]:
    Symbols['open +'].moveTo(*from_)
    Symbols['open +'].lineTo(*to_)


