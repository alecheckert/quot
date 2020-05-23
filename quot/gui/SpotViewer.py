#!/usr/bin/env python
"""
SpotViewer.py -- overlay spots onto a movie after localization

"""
import sys

# File paths
import os 

# Profiling
from time import time 

# Numeric
import numpy as np 

# Dataframes
import pandas as pd 

# Image file reader
from ..read import ImageReader

# Compute trajectory lengths
from ..trajUtils import traj_length 

# Color maps
from matplotlib import cm 
from matplotlib import colors as mpl_colors

# Core GUI utilities
from PySide2.QtCore import Qt 
from PySide2.QtWidgets import QWidget, QLabel, QPushButton, \
    QVBoxLayout, QGridLayout, QApplication, QDialog

# pyqtgraph utilities for images and overlays
from pyqtgraph import ImageView, ScatterPlotItem, GraphItem, \
    PlotWidget, LinearRegionItem
from pyqtgraph.graphicsItems import ScatterPlotItem as SPI_base

# Custom GUI utilities
from .guiUtils import FloatSlider, IntSlider, LabeledQComboBox, \
    set_dark_app, Symbols, keys_to_str, MASTER_COLOR, \
    SingleComboBoxDialog, ImageSubpositionWindow, \
    ImageSubpositionCompare, format_dict 

# Default parameters for spot overlays
pen_width = 3
overlay_params = {'pxMode': False, 'brush': None,
    'pen': {'color': MASTER_COLOR, 'width': pen_width}}

# Colors for conditions
condition_colors = ['#FF5454', '#2DA8FF']

# Custom overlay symbols a little
SPI_base.Symbols['+'] = Symbols['alt +']
SPI_base.Symbols['open +'] = Symbols['open +']
symbol_sizes = {'+': 16.0, 'o': 16.0, 'open +': 20.0}

class SpotViewer(QWidget):
    """
    Overlay localizations or trajectories onto a raw movie.
    Useful for QC.

    The user sees a single image window with the current
    frame and some overlaid localizations.

    Each localization is either colored identically or 
    according to a color scheme. Available color schemes are

        - color each localization by its trajectory index 
            (same color for locs in the same trajectory)

        - color each localization by one of its numerical
            attributes (e.g. intensity, BG, H_det, etc.)

        - color each localization one of two colors according
            to its value for a boolean attribute.

    SpotViewer provides the user a way to create new boolean
    attributes by thresholding attributes, via the "Create
    Boolean attribute" button.

    args
    ----
        image_path          :   str, path to image file (e.g.
                                *.nd2, *.tif, or *.tiff)
        locs_path           :   str, path to CSV with localizations
        gui_size            :   int
        start_frame         :   int, the initial frame shown
        parent              :   root QWidget

    """
    def __init__(self, image_path, locs_path, gui_size=800,
        start_frame=0, parent=None):
        super(SpotViewer, self).__init__(parent=parent)
        self.image_path = image_path 
        self.locs_path = locs_path
        self.gui_size = gui_size 
        self.start_frame = start_frame

        self.initData()
        self.initUI()

    def initData(self):
        """
        Load the datasets for this SpotViewer instance.

        """
        # Make sure the required paths exist
        assert os.path.isfile(self.image_path)
        assert os.path.isfile(self.locs_path)
        assert os.path.splitext(self.image_path)[1] in \
            ['.nd2', '.tif', '.tiff'], \
            "Image file must be one of (*.nd2 *.tif *.tiff)"

        # Make an image file reader
        self.imageReader = ImageReader(self.image_path)

        # Load the set of localizations
        self.locs = pd.read_csv(self.locs_path)

        # Some additional options if these localizations are 
        # assigned a trajectory
        if 'trajectory' in self.locs.columns:

            # Assign the traj_len, if it doesn't already exist
            self.locs = traj_length(self.locs)

            # Keep an array of localizations as an ndarray, for
            # fast indexing when overlaying trajectory histories
            self.tracks = np.asarray(
                self.locs.loc[
                    self.locs['traj_len']>1, 
                    ['frame', 'trajectory', 'y', 'x']
                ].sort_values(by=['trajectory', 'frame'])
            )

            # Adjust for off-by-1/2 plot pixel indexing
            self.tracks[:,2:] = self.tracks[:,2:] + 0.5

        # Generate the set of colors
        self.generate_color_schemes()

        # Assign condition colors (initially all negative)
        self.assign_condition_colors('y')

        # Load the first round of spots
        self.load_image(self.start_frame)

        # Assign each trajectory a color
        self.assign_traj_colors()

        # Assign self.locs_curr, the current set of spots
        self.load_spots(self.start_frame)

    def initUI(self):
        """
        Initialize the user interface.

        """
        # Main window
        self.win = QWidget()

        # Figure out the GUI size
        self.AR = float(self.imageReader.width) / self.imageReader.height
        self.win.resize(self.gui_size*1.5, self.gui_size)
        L_main = QGridLayout(self.win)

        # A subwindow on the left for widgets, and a subwindow
        # on the right for the image
        self.win_right = QWidget(self.win)
        self.win_left = QWidget(self.win)
        L_right = QGridLayout(self.win_right)
        L_left = QGridLayout(self.win_left)
        L_main.addWidget(self.win_right, 0, 1, 1, 3)
        L_main.addWidget(self.win_left, 0, 0, 1, 1)

        ## IMAGES / OVERLAYS

        # ImageView, for rendering image
        self.imageView = ImageView(parent=self.win_right)
        L_right.addWidget(self.imageView, 0, 0)

        # ScatterPlotItem, for overlaying localizations
        self.scatterPlotItem = ScatterPlotItem()
        self.scatterPlotItem.setParentItem(self.imageView.imageItem)

        # GraphItem, for overlaying trajectory histories when
        # desired
        self.graphItem = GraphItem()
        self.graphItem.setParentItem(self.imageView.imageItem)

        # # Make spots clickable
        self.lastClicked = []
        self.scatterPlotItem.sigClicked.connect(self.spot_clicked)

        ## WIDGETS
        widget_align = Qt.AlignTop

        # Frame slider
        self.frame_slider = IntSlider(minimum=0, interval=1, 
            maximum=self.imageReader.n_frames-1, init_value=self.start_frame,
            name='Frame', parent=self.win_left)
        #self.frame_slider.configure(init_value=self.start_frame)
        L_left.addWidget(self.frame_slider, 0, 0, rowSpan=2, 
            alignment=widget_align)
        self.frame_slider.assign_callback(self.frame_slider_callback)

        # Button to toggle spot overlays
        self.B_overlay_state = True 
        self.B_overlay = QPushButton("Overlay spots", parent=self.win_left)
        self.B_overlay.clicked.connect(self.B_overlay_callback)
        L_left.addWidget(self.B_overlay, 1, 0, alignment=widget_align)

        # Button to toggle trajectory trails
        self.B_overlay_trails_state = False 
        self.B_overlay_trails = QPushButton("Overlay histories", parent=self.win_left)
        self.B_overlay_trails.clicked.connect(self.B_overlay_trails_callback)
        L_left.addWidget(self.B_overlay_trails, 1, 1, alignment=widget_align)
        self.B_overlay_trails.stackUnder(self.B_overlay)

        # Menu to select current overlay symbol
        symbol_options = keys_to_str(symbol_sizes.keys())
        self.M_symbol = LabeledQComboBox(symbol_options, "Overlay symbol",
            init_value="o", parent=self.win_left)
        self.M_symbol.assign_callback(self.M_symbol_callback)
        L_left.addWidget(self.M_symbol, 2, 0, alignment=widget_align)

        # Menu to select how the spots are colored
        color_by_options = ["None", "Trajectory", "Quantitative attribute", "Boolean attribute"]
        self.M_color_by = LabeledQComboBox(color_by_options, 
            "Color spots by", init_value="None", parent=self.win_left)
        self.M_color_by.assign_callback(self.M_color_by_callback)
        L_left.addWidget(self.M_color_by, 3, 0, alignment=widget_align)

        # Create a new binary spot condition
        self.B_create_condition = QPushButton("Create Boolean attribute", 
            parent=self.win_left)
        self.B_create_condition.clicked.connect(self.B_create_condition_callback)
        L_left.addWidget(self.B_create_condition, 4, 0, alignment=widget_align)

        # Select the binary column that determines color when "condition"
        # is selected as the color-by attribute
        condition_options = [c for c in self.locs.columns if self.locs[c].dtype == 'bool']
        self.M_condition = LabeledQComboBox(condition_options, "Boolean attribute",
            init_value="None", parent=self.win_left)
        L_left.addWidget(self.M_condition, 3, 1, alignment=widget_align)
        self.M_condition.assign_callback(self.M_condition_callback)

        # Compare spots in a separate window that shows a grid of spots
        self.B_compare_spots = QPushButton("Compare spots", parent=self)
        L_left.addWidget(self.B_compare_spots, 2, 1, alignment=widget_align)
        self.B_compare_spots.clicked.connect(self.B_compare_spot_callback)

        # Some placeholder widgets, for better formatting
        for j in range(6, 18):
            q = QLabel(parent=self.win_left)
            L_left.addWidget(q, j, 0, alignment=widget_align)


        ## DISPLAY
        self.update_image(autoRange=True, autoLevels=True, autoHistogramRange=True)
        self.overlay_spots()
        self.win.show()

    ## CORE FUNCTIONS

    def update_image(self, autoRange=False, autoLevels=False, 
        autoHistogramRange=False):
        """
        Update self.imageView with the current value of self.image
        and change the scatter plots if necessary.

        """
        # Update the image view
        self.imageView.setImage(self.image, autoRange=autoRange, 
            autoLevels=autoLevels, autoHistogramRange=autoHistogramRange)

    def load_spots(self, frame_index=None):
        """
        Load a new set of localizations.

        """
        if frame_index is None:
            frame_index = self.frame_slider.value()

        self.locs_curr = self.locs.loc[self.locs['frame']==frame_index, :]
        self.locs_pos = np.asarray(self.locs_curr[['y', 'x']])
        self.locs_data = self.locs_curr.to_dict(orient='records')

    def load_image(self, frame_index):
        """
        Load a new image.

        """
        self.image = self.imageReader.get_frame(frame_index)

    def change_frame(self, frame_index):
        """
        Load a new frame with a new set of spots.

        """
        # Load the new image
        self.load_image(frame_index)

        # Load the new set of spots
        self.load_spots(frame_index)

    def generate_color_spot_dict(self, color_col):
        """
        From the current set of spots, generate an argument
        suitable to update the scatter plot while coloring 
        each localization by its value of *color_col*. Usually
        *color_col* is either "traj_color" or "attrib_color".

        """
        symbol = self.M_symbol.currentText()
        return [{'pos': tuple(self.locs_pos[i,:]+0.5), 'size': symbol_sizes[symbol],
            'symbol': symbol, 'pen': {'color': self.locs_data[i][color_col], "width": pen_width},
            'brush': None} for i in range(self.locs_pos.shape[0])]

    def overlay_spots(self):
        """
        Overlay the current set of spots onto the current image.

        """
        if self.B_overlay_state:

            # Get the current overlay symbol
            symbol = self.M_symbol.currentText()

            # Get the current coloring scheme
            color_by = self.M_color_by.currentText()

            # Overlay the spots
            if color_by == "None":
                self.scatterPlotItem.setData(pos=self.locs_pos+0.5,
                    data=self.locs_data, symbol=symbol, 
                    size=symbol_sizes[symbol], **overlay_params)
            elif color_by == "Trajectory":
                self.scatterPlotItem.setData(
                    spots=self.generate_color_spot_dict("traj_color"),
                    data=self.locs_data
                )
            elif color_by == "Quantitative attribute": 
                self.scatterPlotItem.setData(
                    spots=self.generate_color_spot_dict("attrib_color"),
                    data=self.locs_data,
                )
            elif color_by == "Boolean attribute":
                self.scatterPlotItem.setData(
                    spots=self.generate_color_spot_dict("condition_color"),
                    data=self.locs_data,
                )
        else:
            self.scatterPlotItem.setData([])

        # Also overlay the trajectory histories, if desired
        if self.B_overlay_trails_state:
            self.overlay_trails()

    def generate_color_schemes(self):
        """
        Generate two color schemes for this dataframe.

        """
        self.n_colors = 133

        # Color scheme 1: viridis
        cmap = cm.get_cmap("viridis", self.n_colors)
        self.colors_0 = np.array(
            [mpl_colors.rgb2hex(cmap(i)[:3]) for i in range(self.n_colors)]
        )

        # Color scheme 2: inferno
        cmap = cm.get_cmap("cool", self.n_colors)
        self.colors_1 = np.array(
            [mpl_colors.rgb2hex(cmap(i)[:3]) for i in range(self.n_colors)]
        )

    def assign_traj_colors(self):
        """
        Assign the "traj_color" column of the dataframe, which assign
        each trajectory a different hex color.

        """
        # If the dataframe does not have a "trajectory" column, then
        # all of the spots get the same color
        if not "trajectory" in self.locs.columns:
            print("assign_traj_colors: `trajectory` column not found in dataframe")
            self.locs["traj_color"] = MASTER_COLOR 

        else:
            self.locs["traj_color"] = self.colors_0[
                self.locs["trajectory"] % self.n_colors]

            # If a spot is not assigned to a trajectory, its color is white
            unassigned = self.locs["trajectory"] == -1 
            self.locs.loc[unassigned, "traj_color"] = "#FFFFFF"

    def assign_attribute_colors(self, attrib):
        """
        Assign the "color" column of the dataframe, which determines
        how individual spots are colored.

        """
        # Get all of the numerical values for this attribute
        x = np.asarray(self.locs[attrib].astype("float64"))
        sanitary = (~np.isinf(x)) & (~np.isnan(x))
        
        # Min and max values for the color map
        cmin = np.percentile(x, 1)
        cmax = np.percentile(x, 95)

        # Divide up the range into bins for each color
        bins = np.linspace(cmin, cmax, self.n_colors-1)

        # Assign each localization to one bin on the basis
        # of its attribute
        assignments = np.zeros(x.shape[0], dtype='int64')
        assignments[sanitary] = np.digitize(x[sanitary], bins=bins)

        # Generate the color indices
        self.locs["attrib_color"] = self.colors_1[assignments]

        # Unsanitary inputs are colored white
        self.locs.loc[~sanitary, "attrib_color"] = "#FFFFFF"

    def assign_condition_colors(self, condition_col):
        """
        Assign the "condition_color" of the dataframe which is 
        keyed to the truth value of *condition_col*.

        """
        indices = self.locs[condition_col].astype('int64')
        self.locs["condition_color"] = condition_colors[0]
        self.locs.loc[self.locs[condition_col].astype('bool'), 
            "condition_color"] = condition_colors[1]

    def overlay_trails(self):
        """
        For each trajectory running in the current frame, overlay
        that trajectory on the current image.

        Relies on an updated self.locs_curr.

        """
        if len(self.locs_curr) == 0:
            self.graphItem.setData()

        elif "trajectory" in self.locs.columns and \
            self.B_overlay_trails_state:

            # Get the current frame index
            frame_index = self.frame_slider.value()

            # Get all spots before or at the current frame
            T = self.tracks[self.tracks[:,0]<=frame_index, :]

            # Find the minimum trajectory index present in this frame
            t = T[T[:,0]==frame_index, 1].min()

            # Get all trajectories that coincide with this frame
            T = T[np.where(T[:,1]==t)[0][0]:,:]

            # Figure out which points correspond to the same
            # trajectory
            indicator = (T[1:,1] - T[:-1,1]) == 0
            connect_indices = np.where(indicator)[0]
            adj = np.asarray([connect_indices, connect_indices+1]).T 

            # Overlay
            self.graphItem.setData(pos=T[:,2:4], adj=adj,
                pen={'color': MASTER_COLOR, 'width': pen_width}, symbol=None,
                brush=None, pxMode=False)
        else:
            self.graphItem.setData()


    ## WIDGET CALLBACKS

    def spot_clicked(self, plot, points):
        """
        Respond to a spot being selected by the user: change
        its color and print all of its associated information
        to the terminal.

        """
        for p in self.lastClicked:
            p.resetPen()

        for p in points:
            print("Spot:")
            print(format_dict(p.data()))
            p.setPen('w', width=3)
        self.lastClicked = points 

    def frame_slider_callback(self):
        """
        Respond to a user change in the frame slider.

        """
        # Get the new frame
        frame_index = self.frame_slider.value()

        # Load a new image and set of localizations
        self.change_frame(frame_index)

        # Update the image
        self.update_image()

        # Update the scatter plot
        self.overlay_spots()

    def B_overlay_callback(self):
        """
        Toggle the spot overlay.

        """
        self.B_overlay_state = not self.B_overlay_state 
        self.overlay_spots()

    def M_color_by_callback(self):
        """
        Change the way that the spots are colored: either None 
        (all the same color), "trajectory", or "attribute".

        """
        color_by = self.M_color_by.currentText()
        if color_by == "Quantitative attribute":

            # Prompt the user to select an attribute to color by
            options = [c for c in self.locs.columns if \
                self.locs[c].dtype in ["int64", "float64", "uint8", \
                    "uint16", "float32", "uint32", "int32"]]
            ex = SingleComboBoxDialog("Attribute", options, 
                init_value="I0", title="Choose attribute color by",
                parent=self)
            ex.exec_()

            # Dialog accepted
            if ex.result() == 1:
                attrib = ex.return_val 

            # Dialog rejected; default to I0
            else:
                print("rejected; defaulting to I0")
                attrib = "I0"

            # Generate the color indices
            self.assign_attribute_colors(attrib)

            # Update the current set of localizations
            self.load_spots(self.frame_slider.value())

        elif color_by == "Boolean attribute":
            self.assign_attribute_colors(self.M_condition.currentText())
            self.overlay_spots()

        else:
            pass 

        # Update the scatter plot
        self.overlay_spots()

    def M_symbol_callback(self):
        """
        Change the spot overlay symbol.

        """
        self.overlay_spots()

    def B_overlay_trails_callback(self):
        """
        Show the history of each trajectory on the raw image.

        """
        self.B_overlay_trails_state = not self.B_overlay_trails_state 

        # Do nothing if these are not trajectories
        if not "trajectory" in self.locs.columns:
            return 

        # Else overlay the trajectory histories
        self.overlay_trails()

    def B_create_condition_callback(self):
        """
        Launch a sub-GUI to generate a Boolean attribute on 
        from one of the spot columns.

        """
        # Prompt the user to create a condition
        ex = ConditionDialog(self.locs, parent=self)
        ex.exec()

        # Accepted - unpack result
        if ex.result() == 1:
            bounds, col = ex.return_val
            l_bound, u_bound = bounds 

            # Add this as an option 
            condition_name = '%s_condition' % col 
            if condition_name not in self.locs.columns:
                self.M_condition.QComboBox.addItems([condition_name])

            # Create a new binary column on the dataframe
            self.locs[condition_name] = np.logical_and(
                self.locs[col]>=l_bound,
                self.locs[col]<=u_bound,
            )
            print("%d/%d localizations in condition %s" % (
                self.locs[condition_name].sum(),
                len(self.locs),
                condition_name
            ))

            # Update plots, assuming that the user wants to see the 
            # result immediately
            self.M_condition.QComboBox.setCurrentText(condition_name)
            self.M_color_by.QComboBox.setCurrentText("Boolean attribute")
            self.assign_condition_colors(condition_name)
            self.load_spots()
            self.overlay_spots()

    def M_condition_callback(self):
        """
        Select the current Boolean attribute to use for determining
        color when using the "Boolean attribute" option in M_color_by.

        """
        # Get the new condition name
        c = self.M_condition.currentText()

        # Assign each localization a color depending on its value
        # with this value
        self.assign_condition_colors(c)

        # Update the plots
        self.load_spots()
        self.overlay_spots()

    def B_compare_spot_callback(self):
        """
        Launch a subwindow that shows a grid of spots in the current
        frame and subsequent frames.

        This callback has two behaviors, depending on the current 
        state of self.M_color_by. If we're color by a Boolean attribute,
        then launch a ImageSubpositionCompare window. Else launch a 
        ImageSubpositionWindow. The former compares two sets of spots
        side-by-side, while the second is just a single grid of spots.

        """
        if self.M_color_by.currentText() == "Boolean attribute":

            # Size of image grid
            N = 8

            # Get the current Boolean attribute column
            col = self.M_condition.currentText()

            # Get as many spots as we can to fill this image grid
            frame_index = self.frame_slider.value()
            positions_false = []
            positions_true = []
            images = []
            while frame_index<self.imageReader.n_frames and \
                (sum([i.shape[0] for i in positions_false])<N**2 or 
                    (sum([i.shape[0] for i in positions_true])<N**2)):

                im = self.imageReader.get_frame(frame_index)
                frame_locs = self.locs.loc[
                    self.locs['frame']==frame_index,
                    ['y', 'x', col],
                ]
                frame_locs_true = np.asarray(frame_locs.loc[
                    frame_locs[col], ['y', 'x']]).astype(np.int64)
                frame_locs_false = np.asarray(frame_locs.loc[
                    ~frame_locs[col], ['y', 'x']]).astype(np.int64)
                images.append(im)
                positions_true.append(frame_locs_true)
                positions_false.append(frame_locs_false)

                frame_index += 1

            # Launch a ImageSubpositionCompare instance
            ex = ImageSubpositionCompare(images, positions_true, positions_false, 
                w=15, N=N, colors=condition_colors, parent=self)

        else:
            # Size of image grid
            N = 10

            # Get as many spots as we can to fill this image grid
            frame_index = self.frame_slider.value()
            positions = []
            images = []
            while sum([i.shape[0] for i in positions])<N**2 and \
                frame_index<self.imageReader.n_frames:

                im = self.imageReader.get_frame(frame_index)
                frame_locs = np.asarray(self.locs.loc[
                    self.locs['frame']==frame_index, ['y', 'x']]).astype(np.int64)
                images.append(im)
                positions.append(frame_locs)
                frame_index += 1

            # Launch an ImageSubpositionWindow
            ex = ImageSubpositionWindow(images, positions,
                w=15, N=N, parent=self)

class ConditionDialog(QDialog):
    """
    Create a new binary column on a set of localizations by 
    drawing a threshold on a 1D histogram of some attribute 
    for those localizations.

    For example, threshold only spots with low intensity (I0)
    or something.

    init
    ----
        locs        :   pandas.DataFrame
        parent      :   root QWidget

    """
    def __init__(self, locs, parent=None):
        super(ConditionDialog, self).__init__(parent=parent)
        self.locs = locs 
        self.initUI()

    def initUI(self):
        """
        Initialize user interface.

        """
        L = QGridLayout(self)
        self.resize(500, 300)

        widget_align = Qt.AlignLeft

        # Available columns for recondition: all numeric columns
        self.columns = list(filter(
            lambda c: self.locs[c].dtype in ['float64', 'float32', \
                'uint8', 'uint16', 'int64'],
            self.locs.columns))

        # For the default, choose `I0` if available; otherwise
        # choose the first column
        init_col = 'I0' if ('I0' in self.columns) else self.columns[0]
        self.load_col(init_col)

        # Main plot
        self.PlotWidget = PlotWidget(name="Create Boolean attribute")
        L.addWidget(self.PlotWidget, 0, 0, alignment=widget_align)

        # Histogram
        self.curve = self.PlotWidget.plot(self.bin_c, self.H, clickable=True)

        # User threshold, as a LinearRegionItem from pyqtgraph
        self.LinearRegion = LinearRegionItem([self.hmin, (self.hmax-self.hmin)*0.25+self.hmin])
        self.PlotWidget.addItem(self.LinearRegion)

        # Drop-down menu to select the column
        self.M_select_col = LabeledQComboBox(self.columns, "Attribute",
            init_value=init_col, parent=self)
        self.M_select_col.assign_callback(self.M_select_col_callback)
        L.addWidget(self.M_select_col, 1, 0, alignment=widget_align)

        # Accept the current threshold
        self.B_accept = QPushButton("Accept", parent=self)
        L.addWidget(self.B_accept, 2, 0, alignment=widget_align)
        self.B_accept.clicked.connect(self.B_accept_callback)

        self.update_histogram()

    def load_col(self, col):
        """
        Get data from a specific column in the locs dataframe.

        args
        ----
            col     :   str

        """
        self.data = np.asarray(self.locs[col])

        # Histogram limits
        self.hmin = self.data.min()
        self.hmax = np.percentile(self.data, 99.9)

        # Binning scheme
        n_bins = 5000
        bin_size = (self.hmax - self.hmin) / n_bins
        self.bin_edges = np.arange(self.hmin, self.hmax, bin_size)

        # Bin the data according to the binning scheme
        self.H, _edges = np.histogram(self.data, bins=self.bin_edges)
        self.bin_c = self.bin_edges[:-1] + (self.bin_edges[1]-self.bin_edges[0])/2.0

    def update_histogram(self):
        """
        Update the main histogram with data from a new column

        """
        self.PlotWidget.clear()
        self.curve = self.PlotWidget.plot(self.bin_c, self.H)
        self.curve.setPen('w')

        # Set default values for linear rect region
        self.LinearRegion.setRegion((self.hmin, np.percentile(self.data, 50)))
        self.PlotWidget.addItem(self.LinearRegion)
        self.curve.updateItems()

    def M_select_col_callback(self):
        """
        Select the current attribute to filter on.

        """
        col = self.M_select_col.currentText()
        self.load_col(col)
        self.update_histogram()

    def B_accept_callback(self):
        """
        Set the return value and exit from the dialog.

        """
        self.return_val = (
            self.LinearRegion.getRegion(),
            self.M_select_col.currentText(),
        )
        self.accept()











