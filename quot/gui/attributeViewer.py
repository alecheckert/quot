#!/usr/bin/env python
"""
AttributeViewer.py -- a dynamic scatter plot of localization
attributes

"""
import sys

# File paths
import os 

# Numeric
import numpy as np 

# Dataframe
import pandas as pd 

# Uniform filter, for computing spatial densities
from scipy.ndimage import uniform_filter 

# Color map
from matplotlib import cm 
from matplotlib import colors as mpl_colors 

# Core GUI utilities
import PySide2
from PySide2.QtCore import Qt 
from PySide2.QtWidgets import QApplication, QLabel, QWidget, \
    QPushButton, QVBoxLayout, QGridLayout, QDialog 

# pyqtgraph plotting utilities
import pyqtgraph as pg 
from pyqtgraph import ScatterPlotItem, GraphicsLayoutWidget

# Custom GUI utilities
from .guiUtils import LabeledQComboBox

class AttributeViewer(QWidget):
    """
    Show a scatter plot of two attributes from a set of localizations.

    init
    ----
        locs_path       :   str, path to a CSV with localization 
                            data
        max_spots       :   int, the maximum number of spots to 
                            plot. It's often too costly to render
                            more than 30000 spots or so
        gui_size        :   int
        parent          :   root QWidget

    """
    def __init__(self, locs_path, max_spots=10000, gui_size=600,
        parent=None):

        super(AttributeViewer, self).__init__(parent=parent)
        self.locs_path = locs_path 
        self.max_spots = max_spots 
        self.gui_size = gui_size 

        self.initData()
        self.initUI()

    def initData(self):
        """
        Load the spots data.

        """
        # Load the locs dataframe
        self.locs = pd.read_csv(self.locs_path).sort_values(by='frame')

        # If too many spots, truncate the dataframe
        if len(self.locs) > self.max_spots:
            print("Truncating dataframe to %d spots" % self.max_spots)
            self.locs = self.locs[:self.max_spots]
        else:
            print("Plotting %d spots" % len(self.locs))

        # List of available attributes to display 
        dtypes = ['float64', 'float', 'int64', 'int32', 'uint16', 'uint8', 'int']
        self.parameters = [c for c in self.locs.columns if \
            self.locs[c].dtype in dtypes]

        # Generate colormap
        self.n_colors = 1000
        self.cmap = cm.get_cmap('viridis', self.n_colors)
        self.cmap_hex = np.array([mpl_colors.rgb2hex(self.cmap(i)[:3]) \
            for i in range(self.n_colors)])

    def initUI(self):
        """
        Initialize the user interface.

        """
        # Main window
        self.win = QWidget()
        L = QGridLayout(self.win)
        self.win.resize(self.gui_size*1.8, self.gui_size)

        # Two subwindows: one on the left for the scatter plot
        # and one on the right for widgets
        self.win_left = QWidget(self.win)
        self.win_right = QWidget(self.win)
        L.addWidget(self.win_left, 0, 0, 1, 3)
        L.addWidget(self.win_right, 0, 3, 1, 1)
        L_left = QGridLayout(self.win_left)
        L_right = QGridLayout(self.win_right)

        # GraphicsLayoutWidget, to organize scatter plot and
        # associated items
        self.graphicsLayoutWidget = GraphicsLayoutWidget(
            parent=self.win_left)

        # PlotItem, to contain the ScatterPlotItem
        self.plotItem = self.graphicsLayoutWidget.addPlot()
        L_left.addWidget(self.graphicsLayoutWidget, 0, 0)

        # ScatterPlotItem, core data display
        self.scatterPlotItem = ScatterPlotItem(symbol='o', 
            brush=None, pxMode=True, pen={'color': '#FFFFFF', 'width': 4.0},
            size=4.0)
        self.plotItem.addItem(self.scatterPlotItem)


        ## WIDGETS
        widget_align = Qt.AlignTop

        # Select parameter to map to the x-axis
        self.M_par_0 = LabeledQComboBox(self.parameters, "x-parameter",
            init_value="x", parent=self.win_right)
        L_right.addWidget(self.M_par_0, 0, 0, alignment=widget_align)
        self.M_par_0.assign_callback(self.M_par_callback)

        # Select parameter to map to the y-axis
        self.M_par_1 = LabeledQComboBox(self.parameters, "y-parameter",
            init_value="y", parent=self.win_right)
        L_right.addWidget(self.M_par_1, 1, 0, alignment=widget_align)
        self.M_par_1.assign_callback(self.M_par_callback)

        # Select which attribute to color the localizations by 
        options = self.parameters + ["density"]
        self.M_color_by = LabeledQComboBox(options, "Color by", 
            init_value="density", parent=self.win_right)
        L_right.addWidget(self.M_color_by, 0, 1, alignment=widget_align)
        self.M_color_by.assign_callback(self.M_color_by_callback)

        # Select the size of the window to use when computing
        # localization density
        window_size_options = [str(j) for j in [3, 5, 7, 9, 11, 13, 15, 19, 23, 31, 41, 61, 81, 101]]
        self.M_density_window = LabeledQComboBox(window_size_options,
            "Density window", init_value="7", parent=self.win_right)
        L_right.addWidget(self.M_density_window, 1, 1, alignment=widget_align)
        self.M_density_window.assign_callback(self.M_density_window_callback)

        # Button to induce a simpler representation that can handle
        # more spots
        self.simple_mode = False 
        self.B_simple = QPushButton("Simple scatter", parent=self.win_right)
        L_right.addWidget(self.B_simple, 2, 0, alignment=widget_align)
        self.B_simple.clicked.connect(self.B_simple_callback)

        # Button to toggle log color scaling
        self.log_scale_mode = True
        self.B_log = QPushButton("Log color scale", parent=self.win_right)
        self.B_log.clicked.connect(self.B_log_callback)
        L_right.addWidget(self.B_log, 2, 1, alignment=widget_align)

        # Empty widgets to manipulate the layout
        n_rows = 15
        for j in range(3, n_rows):
            q = QWidget(self.win_right)
            L_right.addWidget(q, j, 0)

        # Show the main window
        self.update_scatter()
        self.win.show()

    ## CORE FUNCTIONS

    def get_pars(self):
        """
        Get the current attributes mapped to the x and y axes
        of the scatter plot.

        returns
        -------
            (str: x attribute, str: y attribute)

        """
        return self.M_par_0.currentText(), self.M_par_1.currentText()

    def update_scatter(self, rescale=True):
        """
        Update the scatter plot.

        args
        ----
            rescale     :   change axis limits

        """
        cx, cy = self.get_pars()

        # In the special case of plotting x vs y, make sure
        # that the aspect ratio is right
        if ((cx=='x') and (cy=='y')) or ((cx=='y') and (cy=='x')):
            self.plotItem.setAspectLocked(lock=True)
        else:
            self.plotItem.setAspectLocked(lock=False)

        # Update the scatter plots
        if self.simple_mode:
            self.scatter_simple(cx, cy)
        else:
            self.scatter_color(cx, cy)

        # Set axis labels
        labelStyle = {'font-size': '18pt'}
        self.plotItem.setLabel('bottom', text=cx, **labelStyle)
        self.plotItem.setLabel('left', text=cy, **labelStyle)

        # Change axis limits 
        if rescale:
            self.plotItem.autoBtnClicked()

    def scatter_color(self, cx, cy):
        """
        Load the current set of data into a new scatter plot,
        coloring by the current "color by" attribute.

        args
        ----
            cx, cy      :   str, columns in self.locs

        """
        self.scatterPlotItem.setData(size=4.0, brush=None)
        color_attrib = self.M_color_by.currentText()
        if color_attrib == 'density':
            densities, spots = self.make_density(cx, cy)
        else:
            spots = self.make_attrib_colors(color_attrib)
        self.scatterPlotItem.setData(spots=spots)

    def scatter_simple(self, cx, cy):
        """
        A simpler way of representing the data, using black
        and white only.

        args
        ----
            cx, cy  :   str, columns in self.locs

        """
        self.scatterPlotItem.setData(self.locs[cx], self.locs[cy],
            pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 10),
            size=10)

    def make_density(self, c0, c1):
        """
        Get the normalized density of points along the (c0, c1)
        axis in the set of localizations.

        args
        ----
            c0, c1      :   str, columns in self.locs

        returns
        -------
            (
                1D ndarray, the density of each point;
                dict, argument to pass to ScatterPlotItem.setData
            )

        """
        # Get the current density window size
        w = int(self.M_density_window.currentText())

        # Format the desired two attributes as ndarray for 
        # fast indexing
        data = np.asarray(self.locs[[c0, c1]])

        # Each of the attributes could have very different 
        # magnitudes, so scale the distances relative to the 
        # difference between the 5th and 95th percentiles for
        # each attribute.

        # Special case: x and y get the same scaling
        if ((c0=='x') and (c1=='y')) or ((c0=='y') and (c1=='x')):
            ybinsize = 1.0
            xbinsize = 1.0
        else:
            norm = 400.0
            ybinsize = (np.percentile(self.locs[c0], 95) - \
                np.percentile(self.locs[c0], 5)) / norm 
            xbinsize = (np.percentile(self.locs[c1], 95) - \
                np.percentile(self.locs[c1], 5)) / norm        

        # Create the binning scheme from these bin sizes
        ybinedges = np.arange(
            np.percentile(self.locs[c0], 0.1),
            np.percentile(self.locs[c0], 99.9),
            ybinsize
        )
        xbinedges = np.arange(
            np.percentile(self.locs[c1], 0.1),
            np.percentile(self.locs[c1], 99.9),
            xbinsize
        )

        # Bin the data
        H, _yedges, _xedges = np.histogram2d(self.locs[c0], 
            self.locs[c1], bins=(ybinedges, xbinedges))

        # Count the number of points in the neighborhood
        # of each point
        density = uniform_filter(H, w)

        # Digitize the data according to the binning scheme
        # X_int = ((data - np.array([ybinedges.min(), xbinedges.min()])) / \
        #     np.array([ybinsize, xbinsize])).astype('int64')
        # data_y_int = X_int[:,0]
        # data_x_int = X_int[:,1]
        data_y_int = np.digitize(data[:,0], ybinedges)
        data_x_int = np.digitize(data[:,1], xbinedges)

        # Determine which data points fall within the binning
        # scheme
        inside = (data_y_int>=0) & (data_x_int>=0) & \
            (data_y_int<(len(ybinedges)-1)) & \
            (data_x_int<(len(xbinedges)-1))
        data_y_int_inside = data_y_int[inside]
        data_x_int_inside = data_x_int[inside]

        # Retrieve the densities for each point
        point_densities = np.empty(data.shape[0], dtype=density.dtype)
        point_densities[inside] = density[
            data_y_int_inside,
            data_x_int_inside
        ]

        # Set points outside the binning scheme to density 1
        point_densities[~inside] = 1.0 

        # Rescale a little
        point_densities = point_densities * (w**2)
        point_densities[point_densities<=0] = 0.001

        # Log-scale
        if self.log_scale_mode:
            point_densities = np.log(point_densities) / np.log(2.0)
            point_densities[point_densities<0.0] = 0.0

        # Rescale the point densities into color indices
        R = (point_densities*(self.n_colors-1)/point_densities.max()).astype(np.int64)
        R[~inside] = 0
        spot_colors = self.cmap_hex[R]

        # Format the plotting spot dict
        spots = [{'pos': data[i,:], 'pen': {'color': spot_colors[i], 'width': 3.0}} \
            for i in range(data.shape[0])]

        return point_densities, spots 

    def make_attrib_colors(self, attrib):
        """
        Generate a spot format in which the color of each spot
        is keyed to a particular attribute, with appropriate
        rescaling for the different magnitudes of each attribute.

        args
        ----
            attrib  :   str, a column in self.locs

        returns
        -------
            dict, parameters to pass to ScatterPlotItem.setData

        """
        # Current axis attributes
        c0, c1 = self.get_pars()

        # Sort by ascending attribute
        self.locs = self.locs.sort_values(by=attrib)
        XY = np.asarray(self.locs[[c0, c1]])
        Z = np.asarray(self.locs[attrib])

        # Filter out unsanitary values
        sanitary = (~np.isnan(Z)) & (~np.isinf(Z))
        if (~sanitary).any():
            print("Filtering out unsanitary values in %s" % attrib)
        XY = XY[sanitary,:]
        Z = Z[sanitary]

        # Log scale
        if self.log_scale_mode:
            Z[Z<=0.0] = 0.001
            Z = np.log(Z) / np.log(2.0)

        # Bin localizations by their relative value in 
        # the color attribute
        norm = 400.0
        try:
            binsize = (np.percentile(Z, 95) - np.percentile(Z, 5)) / norm 
            binedges = np.arange(np.percentile(Z, 0.01), np.percentile(Z, 99.9), binsize)
            Z[Z<binedges.min()] = binedges.min()
            Z[Z>=binedges.max()] = binedges.max() - binsize
            assignments = np.digitize(Z, bins=binedges).astype('float64')
        except ValueError:
            print("Can't generate a color scale for %s; " \
                "try turning off log scale" % attrib)
            return []

        # Scale into the available color indices
        spot_colors = self.cmap_hex[((assignments * (self.n_colors-1)) / \
            assignments.max()).astype(np.int64)]

        # Generate the argument to ScatterPlotItem.setData
        spots = [{'pos': XY[i,:], 'pen': {'color': spot_colors[i], 'width': 3.0}} \
            for i in range(XY.shape[0])]
        return spots 

    ## WIDGET CALLBACKS

    def M_par_callback(self):
        """
        Change the attribute mapped to the axes.

        """
        self.update_scatter(rescale=True)

    def M_color_by_callback(self):
        """
        Change which attribute is used to generate the color
        scheme for the localizations.

        """
        self.update_scatter(rescale=False)

    def M_density_window_callback(self):
        """
        Change the size of the window used to determine the local
        density of each localization in the current plane.

        """
        if self.M_color_by.currentText() == "density":
            self.update_scatter(rescale=False)

    def B_simple_callback(self):
        """
        Change into simple plotting mode, which uses less memory
        to render the plot but doesn't support color.

        """
        self.simple_mode = not self.simple_mode
        self.update_scatter(rescale=False)

    def B_log_callback(self):
        """
        Toggle between log and linear scales for the color map.

        """
        self.log_scale_mode = not self.log_scale_mode 
        print("Log scale: ", self.log_scale_mode)
        self.update_scatter(rescale=False)





