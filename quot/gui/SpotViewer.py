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
from pyqtgraph import ImageView, ScatterPlotItem, GraphItem 
from pyqtgraph.graphicsItems import ScatterPlotItem as SPI_base

# Custom GUI utilities
from .guiUtils import FloatSlider, IntSlider, LabeledQComboBox, \
    set_dark_app, Symbols, keys_to_str, MASTER_COLOR, \
    SingleComboBoxDialog

# Default parameters for spot overlays
pen_width = 3
overlay_params = {'pxMode': False, 'brush': None,
    'pen': {'color': MASTER_COLOR, 'width': pen_width}}

# Custom overlay symbols a little
SPI_base.Symbols['+'] = Symbols['alt +']
SPI_base.Symbols['open +'] = Symbols['open +']
symbol_sizes = {'+': 16.0, 'o': 16.0, 'open +': 16.0}

class SpotViewer(QWidget):
    def __init__(self, image_path, locs_path, gui_size=800, parent=None):
        super(SpotViewer, self).__init__(parent=parent)
        self.image_path = image_path 
        self.locs_path = locs_path
        self.gui_size = gui_size 

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

        # Load the first round of spots
        self.load_image(0)

        # Assign each trajectory a color
        self.assign_traj_colors()

        # Assign self.locs_curr, the current set of spots
        self.load_spots(0)

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
        # self.lastClicked = []
        # self.scatterPlotItem.sigClicked.connect(self.spot_clicked)

        ## WIDGETS
        widget_align = Qt.AlignTop

        # Frame slider
        self.frame_slider = IntSlider(minimum=0, interval=1, 
            maximum=self.imageReader.n_frames-1, init_value=0,
            name='Frame', parent=self.win_left)
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
        self.B_overlay_trails = QPushButton("Show history", parent=self.win_left)
        self.B_overlay_trails.clicked.connect(self.B_overlay_trails_callback)
        L_left.addWidget(self.B_overlay_trails, 2, 0, alignment=widget_align)
        self.B_overlay_trails.stackUnder(self.B_overlay)

        # Menu to select current overlay symbol
        symbol_options = keys_to_str(symbol_sizes.keys())
        self.M_symbol = LabeledQComboBox(symbol_options, "Overlay symbol",
            init_value="o", parent=self.win_left)
        self.M_symbol.assign_callback(self.M_symbol_callback)
        L_left.addWidget(self.M_symbol, 3, 0, alignment=widget_align)

        # Menu to select how the spots are colored
        color_by_options = ["None", "trajectory", "attribute"]
        self.M_color_by = LabeledQComboBox(color_by_options, 
            "Color spots by", init_value="None", parent=self.win_left)
        self.M_color_by.assign_callback(self.M_color_by_callback)
        L_left.addWidget(self.M_color_by, 4, 0, alignment=widget_align)

        # Create a new binary spot condition
        self.B_create_condition = QPushButton("Create new condition", 
            parent=self.win_left)
        self.B_create_condition.clicked.connect(self.B_create_condition_callback)
        L_left.addWidget(self.B_create_condition, 5, 0, alignment=widget_align)

        # Some placeholder widgets, for better formatting
        for j in range(6, 15):
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

    def load_spots(self, frame_index):
        """
        Load a new set of localizations.

        """
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
            elif color_by == "trajectory":
                self.scatterPlotItem.setData(
                    spots=self.generate_color_spot_dict("traj_color"),
                    data=self.locs_data
                )
            elif color_by == "attribute": 
                self.scatterPlotItem.setData(
                    spots=self.generate_color_spot_dict("attrib_color"),
                    data=self.locs_data,
                )
            elif color_by == "condition":
                pass 
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
        if color_by == "attribute":

            # Prompt the user to select an attribute to color by
            options = [c for c in self.locs.columns if \
                self.locs[c].dtype in ["int64", "float64"]]
            ex = SingleComboBoxDialog("Attribute", options, 
                init_value="I0", title="Choose attribute color by",
                parent=self)
            if ex.exec_() is QDialog.Accepted:
                attrib = ex.return_val 
            else:
                attrib = "I0"

            # Generate the color indices
            self.assign_attribute_colors(attrib)

            # Update the current set of localizations
            self.load_spots(self.frame_slider.value())

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
        Launch a sub-GUI to generate a binary condition on 
        one of the spot attributes.

        """
        pass 



if __name__ == '__main__':
    app = QApplication([])
    set_dark_app(app)
    x = SpotViewer(
        '78203_BioPipeline_Run1_20200508_222010_652__Plate000_WellB19_ChannelP95_Seq0035.nd2',
        '78203_BioPipeline_Run1_20200508_222010_652__Plate000_WellB19_ChannelP95_Seq0035_tracks.csv',
        #'test_without_traj.csv',
    )
    sys.exit(app.exec_())









