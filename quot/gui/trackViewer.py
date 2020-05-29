#!/usr/bin/env python
"""
trackViewer.py -- run tracking while modifying tracking
settings in real-time

"""
import sys

# File paths
import os

# Numeric
import numpy as np 

# DataFrames
import pandas as pd 

# quot image reader
from ..read import ImageReader 

# quot tracking utilities
from ..track import track 

# Color palette
from matplotlib import cm 
from matplotlib import colors as mpl_colors 

# Core GUI utilities
from PySide2.QtCore import Qt 
from PySide2.QtWidgets import QWidget, QLabel, QPushButton, \
    QVBoxLayout, QGridLayout, QDialog 

# pyqtgraph image display utilities
from pyqtgraph import ImageView, ScatterPlotItem, GraphItem 
from pyqtgraph.graphicsItems import ScatterPlotItem as SPI_base

# Custom GUI utilities
from .guiUtils import IntSlider, FloatSlider, LabeledQComboBox, \
    Symbols, keys_to_str, getTextInputs, PromptSelectROI

# Config settings
from ..read import read_config 
config_path = os.path.join(os.path.dirname(__file__), "CONFIG.toml")
CONFIG = read_config(config_path)
TRACK_SLIDER_CONFIG = CONFIG['track_slider_config']

# Custom spot overlay symbols
SPI_base.Symbols['+'] = Symbols['alt +']
SPI_base.Symbols['open +'] = Symbols['open +']
symbol_sizes = {'+': 4.0, 'o': 4.0, 'open +': 4.0}

# Global parameters for plotting
pen_width = 3.0

class TrackViewer(QWidget):
    def __init__(self, image_path, locs_path, pixel_size_um=0.16,
        frame_interval=0.00548, gui_size=600, cmap='plasma',
        start_frame=0, stop_frame=500, parent=None):

        super(TrackViewer, self).__init__(parent=parent)
        self.image_path = image_path 
        self.locs_path = locs_path 
        self.pixel_size_um = pixel_size_um 
        self.frame_interval = frame_interval 
        self.gui_size = gui_size 
        self.cmap = cmap 
        self.start_frame = start_frame
        self.stop_frame = stop_frame

        self.initData()
        self.initUI()

    def initData(self):

        # Load the set of localizations
        self.all_locs = pd.read_csv(self.locs_path)

        # Adjust the positions for off-by-1/2 pixel errors
        self.all_locs[['y', 'x']] = self.all_locs[['y', 'x']] + 0.5

        # Load the images over which to plot localizations
        self.imageReader = ImageReader(self.image_path)

        # Set the default subregion to the full field of view
        self.subregion_kwargs = self.imageReader._process_subregion()

        # Adjust frame limits
        self.stop_frame = min(self.stop_frame, self.imageReader.n_frames-1)

        # Get the first image
        self.image = self.imageReader.get_frame(self.start_frame)

        # Set the number of overlay graphs. Due to the structure
        # of pyqtgraph, overlay is fastest when we have separate
        # graphs for each color of trajectory.
        self.n_graphs = 20
        self.modulus = 7691
        cmap = cm.get_cmap("plasma", self.n_graphs)
        self.graph_colors = np.array(
            [mpl_colors.rgb2hex(cmap(i)[:3]) for i in range(self.n_graphs)]
        )

        # Assign each trajectory to one of the graphs. We'll do
        # this each time we retrack the localizations
        self.locs = self.all_locs.loc[
            np.logical_and(
                self.all_locs['frame']>=self.start_frame,
                self.all_locs['frame']<=self.stop_frame,
            ), :
        ].copy()
        self.hash_tracks()

    def initUI(self):

        self.win = QWidget()
        L_master = QGridLayout(self.win)

        # Left window, for image
        win_left = QWidget(self.win)
        L_left = QGridLayout(win_left)
        L_master.addWidget(win_left, 0, 0, 1, 2)

        # Right window, for widgets
        win_right = QWidget(self.win)
        L_right = QGridLayout(win_right)
        L_master.addWidget(win_right, 0, 2, 1, 1)

        ## IMAGE / TRAJECTORY OVERLAY

        # ImageView, for showing image
        self.imageView = ImageView(parent=win_left)
        L_left.addWidget(self.imageView, 0, 0)
        self.imageView.setImage(self.image)

        # GraphItems, for overlaying trajectories in various 
        # colors
        self.graphItems = [GraphItem() for j in range(self.n_graphs)]
        for j in range(self.n_graphs):
            self.graphItems[j].setParentItem(self.imageView.imageItem)
            self.graphItems[j].scatter.setPen(color=self.graph_colors[j], width=4.0)

        # ScatterPlotItem, for occasional overlay of search radii
        self.scatterPlotItem = ScatterPlotItem()
        self.scatterPlotItem.setParentItem(self.imageView.imageItem)

        ## WIDGETS

        widget_align = Qt.AlignTop

        # Frame slider
        self.frame_slider = IntSlider(minimum=self.start_frame, interval=1, 
            maximum=self.stop_frame, init_value=self.start_frame,
            name='Frame', parent=win_right)
        self.frame_slider.assign_callback(self.frame_slider_callback)
        L_right.addWidget(self.frame_slider, 0, 0, alignment=widget_align)

        # Button to toggle trajectory overlay
        self.B_overlay_state = True 
        self.B_overlay = QPushButton("Overlay trajectories", parent=win_right)
        self.B_overlay.clicked.connect(self.B_overlay_callback)
        L_right.addWidget(self.B_overlay, 7, 0, alignment=widget_align)

        # Menu to select the tracking option
        methods = keys_to_str(TRACK_SLIDER_CONFIG.keys())
        self.M_method = LabeledQComboBox(methods, "Method",
            init_value="diffusion", parent=win_right)
        L_right.addWidget(self.M_method, 1, 0, alignment=widget_align)
        self.M_method.assign_callback(self.M_method_callback)

        # Five semantically-flexible FloatSliders to set tracking 
        # method parameters
        self.floatSliders = []
        self.n_sliders = 5
        for j in range(self.n_sliders):
            slider = FloatSlider(parent=win_right, min_width=100)
            L_right.addWidget(slider, 2+j, 0, alignment=widget_align)
            slider.assign_callback(getattr(self, "slider_%d_callback" % j))
            self.floatSliders.append(slider)

        # Button to change ROI
        self.B_change_roi = QPushButton("Change ROI", parent=win_right)
        L_right.addWidget(self.B_change_roi, 8, 0, alignment=widget_align)
        self.B_change_roi.clicked.connect(self.B_change_roi_callback)

        # Button to change frame limits
        self.B_frame_limits = QPushButton("Change frame limits", parent=win_right)
        L_right.addWidget(self.B_frame_limits, 9, 0, alignment=widget_align)
        self.B_frame_limits.clicked.connect(self.B_frame_limits_callback)

        # Button to toggle search radius overlay
        self.B_search_radius_state = False 
        self.B_search_radius = QPushButton("Show search radii", parent=win_right)
        L_right.addWidget(self.B_search_radius, 9, 1, alignment=widget_align)
        self.B_search_radius.clicked.connect(self.B_search_radius_callback)

        # overlay the first set of localizations
        self.change_track_method(self.M_method.currentText())
        self.retrack()
        self.hash_tracks()
        self.update_tracks()

        # Resize and show GUI
        self.win.resize(self.gui_size*2, self.gui_size)
        self.win.show()

    ## CORE FUNCTIONS

    def set_track_params(self, reset=False, **kwargs):
        """
        Modify or reset the current set of tracking parameters.

        args
        ----
            reset       :   bool. If True, reset the set of 
                            tracking parameters back to defaults
                            for the current tracking method
            **kwargs    :   kwargs to the current tracking method

        """
        if reset:
            method = self.M_method.currentText()
            self.track_params = {
                'method': method,
                'pixel_size_um': self.pixel_size_um,
                'frame_interval': self.frame_interval,
                **kwargs
            }
            for k in [j for j in TRACK_SLIDER_CONFIG[method].keys() if j not in kwargs.keys()]:
                kwargs[k] = TRACK_SLIDER_CONFIG[method][k]['init_value']
        else:
            for k in kwargs.keys():
                self.track_params[k] = kwargs[k]

    def update_image(self, frame_index=None, autoRange=False, autoLevels=False,
        autoHistogramRange=False):
        """
        Get a new frame from the reader and update the 
        central ImageView.

        args
        ----
            frame_index     :   int, desired frame
            autoRange       :   bool, reset the ImageView window
            autoLevels      :   bool, reset LUTs
            autoHistogramRange  :   bool, reset LUT selection window

        """
        if frame_index is None:
            frame_index = self.frame_slider.value()
        self.image = self.imageReader.get_subregion(frame_index, **self.subregion_kwargs)
        self.imageView.setImage(self.image, autoRange=autoRange,
            autoLevels=autoLevels, autoHistogramRange=autoHistogramRange)

    def load_tracks_in_frame_limits(self):
        """
        Given the current set of frame limits and/or ROI 
        limits, take only localizations in the frame limits/
        ROI and assign them to self.locs.

        """
        # Take only localizations in the ROI
        self.locs = self.all_locs.loc[
            (
                np.array(self.all_locs['frame']>=self.frame_slider.minimum) & \
                np.array(self.all_locs['frame']<=self.frame_slider.maximum) & \
                np.array(self.all_locs['y']>=self.subregion_kwargs['y0']) & \
                np.array(self.all_locs['y']<=self.subregion_kwargs['y1']) & \
                np.array(self.all_locs['x']>=self.subregion_kwargs['x0']) & \
                np.array(self.all_locs['x']<=self.subregion_kwargs['x1'])
            ), :
        ].copy()

        # Adjust for the start pos
        self.locs['y'] = self.locs['y'] - self.subregion_kwargs['y0']
        self.locs['x'] = self.locs['x'] - self.subregion_kwargs['x0']

    def hash_tracks(self):
        """
        Save the current value of self.locs as an ndarray with 
        the name self.tracks. This facilitates much more rapid
        indexing, since ndarrays have faster operations than 
        pandas.DataFrames.

        This also hashes the trajectory index to a graph index
        that determines in which color the trajectory will render.

        """
        self.locs['graph_index'] = (self.locs['trajectory'] * self.modulus) % self.n_graphs
        self.tracks = np.asarray(self.locs[
            ['frame', 'trajectory', 'y', 'x', 'graph_index']
        ].sort_values(by=['trajectory', 'frame']))

    def overlay_search_radii(self):
        """
        On top of each localization in the current frame, overlay
        the search radii.

        """
        if self.B_search_radius_state:

            # Get all localizations in the current frame
            frame_index = self.frame_slider.value()
            pos = self.tracks[self.tracks[:,0]==frame_index, 2:4]

            if 'search_radius' in self.track_params.keys() and \
                'pixel_size_um' in self.track_params.keys():
                sr_pxl = self.track_params['search_radius']/self.track_params['pixel_size_um']
                self.scatterPlotItem.setData(
                    pos=pos, symbol='o', pxMode=False,
                    pen={'color': '#FFFFFF', 'width': 2},
                    size=sr_pxl*2,
                    brush=None,
                )
            else:
                self.scatterPlotItem.setData([])
        else:
            self.scatterPlotItem.setData([])

    def retrack(self):
        """
        Run tracking on the set of localizations with the current
        settings.

        """
        self.locs = track(self.locs, **self.track_params)

    def change_track_method(self, method):
        """
        Change the current tracking method, updating the sliders
        etc.

        """
        # Reset the tracking params
        self.set_track_params(reset=True, method=method)

        # Reconfigure the sliders
        n_params = len(TRACK_SLIDER_CONFIG[method])
        for j, name in enumerate(TRACK_SLIDER_CONFIG[method].keys()):
            self.floatSliders[j].show()
            self.floatSliders[j].configure(
                minimum=TRACK_SLIDER_CONFIG[method][name]['minimum'],
                maximum=TRACK_SLIDER_CONFIG[method][name]['maximum'],
                init_value=TRACK_SLIDER_CONFIG[method][name]['init_value'],
                name=name,
                interval=TRACK_SLIDER_CONFIG[method][name]['interval'],
                return_int=(TRACK_SLIDER_CONFIG[method][name]['type']=='int'),
            )
        for j in range(n_params, len(self.floatSliders)):
            self.floatSliders[j].hide()

        # Retrack and render localizations
        self.load_tracks_in_frame_limits()
        self.retrack()
        self.hash_tracks()
        self.update_tracks()

    def update_tracks(self):
        """
        Get the current set of trajectories from self.tracks
        and overlay onto the raw image.

        """
        if self.B_overlay_state:

            # Get the current frame index
            frame_index = self.frame_slider.value()

            # Get all spots before or in the current frame
            T = self.tracks[self.tracks[:,0]<=frame_index, :]

            # Find the minimum trajectory index present in this frame
            try:
                t = T[T[:,0]==frame_index, 1].min()
            except ValueError:
                for gi in range(self.n_graphs):
                    self.graphItems[gi].setData(pen=None)
                return 

            # Get all trajectories that coincide with this frame
            T = T[np.where(T[:,1]==t)[0][0]:,:]

            # Iterate over each graph group
            for gi in range(self.n_graphs):

                # Take the trajectories that correspond to this graph
                G = T[T[:,4]==gi, :]

                # No points left - hide everything
                if len(G.shape)<2 or G.shape[0] < 1:
                    self.graphItems[gi].setData(
                        pen=None
                    )

                else:
                    # Figure out which points correspond to the same trajectory
                    indicator = (G[1:,1] - G[:-1,1]) == 0
                    connect_indices = np.where(indicator)[0]
                    adj = np.asarray([connect_indices, connect_indices+1]).T 

                    # Perform the overlay 
                    self.graphItems[gi].setData(
                        pos=G[:,2:4], 
                        adj=adj,
                        pen={'color': self.graph_colors[gi], 'width': pen_width},
                        symbol='o', 
                        #symbolPen={'color': self.graph_colors[gi], 'width': pen_width},
                        brush=None,
                        pxMode=False, 
                        size=symbol_sizes['o']
                    )

        else:
            for gi in range(self.n_graphs):
                self.graphItems[gi].setData(pen=None)

    ## WIDGET CALLBACKS

    def frame_slider_callback(self):
        """
        Respond to a user change in the frame slider.

        """
        frame_index = self.frame_slider.value()
        self.update_image(frame_index=frame_index)
        self.update_tracks()
        self.overlay_search_radii()

    def B_overlay_callback(self):
        """
        Toggle the trajectory overlay.

        """
        self.B_overlay_state = not self.B_overlay_state
        self.update_tracks()

    def M_method_callback(self):
        """
        Change the current tracking method.

        """
        m = self.M_method.currentText()
        self.change_track_method(m)
        self.retrack()
        self.hash_tracks()
        self.update_tracks()

    def slider_callback(self, slider_index):
        """
        Respond to a user change in one of the sliders that 
        sets tracking parameters.

        """
        # Get the identity of the slider that changed
        name = self.floatSliders[slider_index].name 

        # Get the new value of this slider
        value = self.floatSliders[slider_index].value()

        # Set this value in the tracking params
        self.track_params[name] = value 

        # Retrack and rerender
        self.retrack()
        self.hash_tracks()
        self.update_tracks()

        # Special case: if this slider is the search radius,
        # also change the search radius icon
        if name == 'search_radius':
            self.overlay_search_radii()

    def slider_0_callback(self):
        self.slider_callback(0)

    def slider_1_callback(self):
        self.slider_callback(1)

    def slider_2_callback(self):
        self.slider_callback(2)

    def slider_3_callback(self):
        self.slider_callback(3)

    def slider_4_callback(self):
        self.slider_callback(4)

    def B_change_roi_callback(self):
        """
        Change the current ROI used for tracking.

        """
        # Make a maximum intensity projection
        if not hasattr(self, "max_int_proj"):
            print("Projecting...")
            self.max_int_proj = self.imageReader.max_int_proj()

        # Prompt the user to choose an ROI
        y_slice, x_slice = PromptSelectROI(
            self.max_int_proj, parent=self)

        # If the user has accepted, reset the ROI
        if not y_slice is None:
            y0 = y_slice.start
            y1 = y_slice.stop
            x0 = x_slice.start 
            x1 = x_slice.stop 
            self.subregion_kwargs = {'y0': y0, 'y1': y1,
                'x0': x0, 'x1': x1}

        # Reslice the locs
        self.update_image(autoRange=True, autoLevels=True,
            autoHistogramRange=True)
        self.load_tracks_in_frame_limits()
        self.retrack()
        self.hash_tracks()
        self.update_tracks()
        self.overlay_search_radii()

    def B_frame_limits_callback(self):
        """
        Change the current frame limits.

        """
        # Prompt the user to select a start and stop frame
        labels = [
            'Start frame (min 0)', 
            'Stop frame (max %d)' % (self.imageReader.n_frames-1)
        ]
        defaults = [
            self.start_frame,
            self.stop_frame
        ]
        choices = getTextInputs(labels, defaults, 
            title="Select frame limits")

        # Reconfigure the slider
        self.frame_slider.configure(
            minimum=int(choices[0]),
            maximum=int(choices[1]),
            interval=1,
            name='Frame',
            init_value=int(choices[0]),
        )

        # Take the tracks in the frame limits
        self.load_tracks_in_frame_limits()

        # Retrack etc.
        self.update_image(frame_index=int(choices[0]),
            autoRange=True, autoLevels=True, 
            autoHistogramRange=True)
        self.retrack()
        self.hash_tracks()
        self.update_tracks()
        self.overlay_search_radii()

    def B_search_radius_callback(self):
        """
        Toggle overlaying the search radius onto the raw image.

        """
        self.B_search_radius_state = not self.B_search_radius_state
        self.overlay_search_radii()






