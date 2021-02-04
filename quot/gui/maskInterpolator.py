#!/usr/bin/env python
"""
maskInterpolator.py -- a graphic user interface for drawing masks
defined at a discrete set of frames, then interpolating the shape
of the mask at intermediate frames.

"""
import sys

# File paths
import os

# Numeric
import numpy as np 
from scipy import ndimage as ndi

# Dataframes
import pandas as pd 

# Path object, for fast determination whether points lie inside or
# outside a closed path
from matplotlib.path import Path 

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# File readers
from ..read import ImageReader

# Convert a binary 2D mask to a 2D image of mask edges
from ..helper import (
    get_edges,
    get_ordered_mask_points
)

# Mask interpolator
from quot.mask import MaskInterpolator as QuotMaskInterpolator

# Core GUI utilities
import PySide2
from PySide2 import QtCore
from PySide2.QtCore import Qt 
from PySide2.QtWidgets import QWidget, QLabel, QPushButton, \
    QVBoxLayout, QGridLayout, QApplication, QDialog 

# pyqtgraph plotting utilities
from pyqtgraph import ImageView, ScatterPlotItem, PlotWidget

# Custom GUI utilities
from .guiUtils import (
    IntSlider,
    getSaveFilePath,
    getOpenFilePath,
    getTextInputs
)

# Mask selector
from .masker import Masker 

class MaskInterpolator(QWidget):
    """
    Define masks based on an image file for two frames, then interpolate
    the masks for the intermediate frames. Also includes options to apply
    these masks to a target file.

    """
    def __init__(self, parent=None):
        super(MaskInterpolator, self).__init__(parent=parent)
        self.initUI()

    def initUI(self):

        # Main layout
        self.win = QWidget()
        L_master = QGridLayout(self.win)
        self.win.setWindowTitle("Mask interpolator")

        # Widget alignment
        widget_align = Qt.AlignLeft

        # Display the masks, when defined
        self.plotWidget = PlotWidget()
        L_master.addWidget(self.plotWidget, 0, 0, 1, 2, alignment=widget_align)

        # Button to select the first mask file
        self.B_select_mask_0 = QPushButton("Choose mask channel file 1", parent=self.win)
        self.B_select_mask_0.clicked.connect(self.B_select_mask_0_callback)
        L_master.addWidget(self.B_select_mask_0, 1, 0, 1, 1, alignment=widget_align)

        # Show the currently selected mask file 1
        self.mask_file_0_label = QLabel(parent=self.win)
        L_master.addWidget(self.mask_file_0_label, 2, 0, 1, 1, alignment=widget_align)

        # Generate the first set of masks
        self.B_generate_masks_0 = QPushButton("Draw mask 1", parent=self.win)
        self.B_generate_masks_0.clicked.connect(self.B_generate_masks_0_callback)
        L_master.addWidget(self.B_generate_masks_0, 3, 0, 1, 1, alignment=widget_align)

        # Alternatively, load masks from an existing file
        self.B_load_mask_0 = QPushButton("Load mask 1", parent=self.win)
        self.B_load_mask_0.clicked.connect(self.B_load_mask_0_callback)
        L_master.addWidget(self.B_load_mask_0, 4, 0, 1, 1, alignment=widget_align)

        # Button to select the second mask file
        self.B_select_mask_1 = QPushButton("Choose mask channel file 2", parent=self.win)
        self.B_select_mask_1.clicked.connect(self.B_select_mask_1_callback)
        L_master.addWidget(self.B_select_mask_1, 5, 0, 1, 1, alignment=widget_align)

        # Show the currently selected mask file 2
        self.mask_file_1_label = QLabel(parent=self.win)
        L_master.addWidget(self.mask_file_1_label, 6, 0, 1, 1, alignment=widget_align)

        # Generate the first set of masks
        self.B_generate_masks_1 = QPushButton("Draw mask 2", parent=self.win)
        self.B_generate_masks_1.clicked.connect(self.B_generate_masks_1_callback)
        L_master.addWidget(self.B_generate_masks_1, 7, 0, 1, 1, alignment=widget_align)       

        # Alternatively, load masks from an existing file
        self.B_load_mask_1 = QPushButton("Load mask 2", parent=self.win)
        self.B_load_mask_1.clicked.connect(self.B_load_mask_1_callback)
        L_master.addWidget(self.B_load_mask_1, 8, 0, 1, 1, alignment=widget_align)

        # Apply the set of masks to a file
        self.B_apply = QPushButton("Apply masks", parent=self)
        self.B_apply.clicked.connect(self.B_apply_callback)
        L_master.addWidget(self.B_apply, 9, 0, 1, 1, alignment=widget_align)

        # Show the main window
        self.win.show()

    ## CORE FUNCTIONS

    def get_currdir(self):
        """
        Get the last directory that was accessed by the user. If 
        no directory was previously accessed, then this is just 
        the same directory as the image file.

        """
        if not hasattr(self, "_currdir"):
            self.set_currdir(os.getcwd())
        return self._currdir

    def set_currdir(self, path):
        """
        Set the directory returned by self.get_currdir().

        args
        ----
            path            :   str, a file path or directory path. If 
                                a file path, its parent directory is used.

        """
        if os.path.isfile(path):
            self._currdir = os.path.split(os.path.abspath(path))[0]
        elif os.path.isdir(path):
            self._currdir = path

    def update_label(self, text, label=0):
        """
        args
        ----
            text        :   str, text to display in the QLabel
            label       :   0 or 1, corresponding to 
                            self.mask_file_0_label or 
                            self.mask_file_1_label respectively

        """
        if label == 0:
            self.mask_file_0_label.setText(text)
        elif label == 1:
            self.mask_file_1_label.setText(text)

    def select_mask_file(self, index):
        """
        Prompt the user to select an image file to user for masking.

        args
        ----
            index       :   int, either 0 or 1. If 0, the first mask 
                            file is selected and otherwise the second
                            mask is selected

        """
        path = getOpenFilePath(self.win, "Select masking channel file %d" % (index+1),
            "Image files (*.tif *.tiff *.nd2)",
            initialdir=self.get_currdir())
        self.set_currdir(path)
        self.update_label("{} (no masks defined)".format(path), label=index)
        setattr(self, "mask_path_{}".format(index), path)

        # Delete previously defined masks, if any
        if hasattr(self, "mask_edges_{}".format(index)):
            delattr(self, "mask_edges_{}".format(index))

    def load_mask(self, index, trial_interpolate=False, plot=False):
        """
        Load preexisting masks from a file.

        """
        # Prompt the user to enter a mask definitions file
        path = getOpenFilePath(self.win, "Select mask definitions %d" % (index+1),
            "Mask edge or image files (*.csv *.tif *.tiff *.nd2)",
            initialdir=self.get_currdir())
        self.set_currdir(path)

        # Delete the previously defined masks, if any
        if hasattr(self, "mask_path_{}".format(index)):
            delattr(self, "mask_path_{}".format(index))
        if hasattr(self, "mask_edges_{}".format(index)):
            delattr(self, "mask_edges_{}".format(index))

        # Calculate edges from this mask
        ext = os.path.splitext(path)[-1]
        if ext == ".csv":
            try:
                df = pd.read_csv(path)
                for c in ["filename", "mask_index", "y", "x", "vertex"]:
                    assert c in df.columns
            except:
                print("File {} not in the correct format; must be a CSV with " \
                    "the 'filename', 'mask_index', 'y', 'x', and 'vertex' " \
                    "columns".format(path))
                return 

            # Take only the first mask encountered
            mask_indices = df["mask_index"].unique()
            points = [np.asarray(df.loc[df["mask_index"] == mask_indices[0], ["y", "x"]])]

        elif ext in [".tif", ".tiff", ".nd2"]:
            with ImageReader(path) as f:
                mask_im = f.get_frame(0)
            edges = get_edges(mask_im>0)
            points = [get_ordered_mask_points(edges, max_points=200)]           

        setattr(self, "mask_edges_{}".format(index), points) 
        setattr(self, "mask_path_{}".format(index), path)
        self.update_label("{} ({} masks defined)".format(path, len(points)), index)

        # Show a trial interpolation (only if the other 
        # mask is also defined)
        if trial_interpolate and hasattr(self, "mask_edges_{}".format(1-index)):
            interpolator = QuotMaskInterpolator(
                [self.mask_edges_0[0], self.mask_edges_1[0]],
                [0, 100],
                n_vertices=101,
                interp_kind="linear",
                plot=False,
            )
            self.plot_mask_edges_interpolated(interpolator)

        elif plot:
            edges_0 = getattr(self, "mask_edges_0") if hasattr(self, "mask_edges_0") else None
            edges_1 = getattr(self, "mask_edges_1") if hasattr(self, "mask_edges_1") else None
            self.plot_mask_edges(edges_0=edges_0, edges_1=edges_1)

    def generate_masks(self, index, plot=False, trial_interpolate=False):
        """
        Prompt the user to define some masks. 

        args
        ----
            index       :   int, either 0 or 1. If 0, generate masks from
                            self.mask_path_0. If 1, generate masks from 
                            self.mask_path_1. 
            plot        :   bool, update the plot with the new masks
            trial_interpolate   :   bool, show a trial interpolation

        """
        # If the user has not yet selected a mask file, bail
        if not hasattr(self, "mask_path_{}".format(index)):
            print("Mask file {} not selected".format(index))
            return 
        else:
            path = getattr(self, "mask_path_{}".format(index))

        # Prompt the user to enter some masks
        ex = Masker(path, dialog_mode=True, parent=self.win)

        # If the user accepts the definitions, save
        if ex.exec_() == QDialog.Accepted:
            setattr(self, "mask_edges_{}".format(index), ex.return_val)
            self.update_label("{} ({} masks defined)".format(path, len(ex.return_val)), index)

            # Show a trial interpolation (only if the other 
            # mask is also defined)
            if trial_interpolate and hasattr(self, "mask_edges_{}".format(1-index)):
                interpolator = QuotMaskInterpolator(
                    [self.mask_edges_0[0], self.mask_edges_1[0]],
                    [0, 100],
                    n_vertices=101,
                    interp_kind="linear",
                    plot=False,
                )
                self.plot_mask_edges_interpolated(interpolator)

            elif plot:
                edges_0 = getattr(self, "mask_edges_0") if hasattr(self, "mask_edges_0") else None
                edges_1 = getattr(self, "mask_edges_1") if hasattr(self, "mask_edges_1") else None
                self.plot_mask_edges(edges_0=edges_0, edges_1=edges_1)

        else:
            print("Dialog canceled")

    def plot_mask_edges(self, edges_0=None, edges_1=None):
        """
        Plot a set of mask edges on the raw image.

        args
        ----
            edges_0         :   list of 2D ndarray of shape (n_points, 2),
                                the YX coordinates of each mask in the first
                                set
            edges_1         :   list of 2D ndarray of shape (n_points, 2),
                                the YX coordinates of each mask in the second
                                set

        """
        self.plotWidget.clear()
        if not edges_0 is None:
            for mask_edge in edges_0:
                M = np.concatenate((mask_edge, np.array([mask_edge[0,:]])), axis=0)
                self.plotWidget.plot(M[:,0], M[:,1], pen=(127, 253, 186))
        if not edges_1 is None:
            for mask_edge in edges_1:
                M = np.concatenate((mask_edge, np.array([mask_edge[0,:]])), axis=0)
                self.plotWidget.plot(M[:,0], M[:,1], pen=(36, 255, 28))           

    def plot_mask_edges_interpolated(self, quot_mask_interpolator):
        """
        Plot the interpolated set of mask edges.

        args
        ----
            quot_mask_interpolated      :   quot.mask.MaskInterpolator

        """
        E = quot_mask_interpolator.mask_edges
        mask_edges_0 = [E[:,0,:]]
        mask_edges_1 = [E[:,1,:]]
        self.plot_mask_edges(edges_0=mask_edges_0, edges_1=mask_edges_1)
        for j in range(E.shape[0]):
            self.plotWidget.plot(
                [mask_edges_0[0][j,0], mask_edges_1[0][j,0]],
                [mask_edges_0[0][j,1], mask_edges_1[0][j,1]],
                pen=(255, 165, 28)
            )

    def apply_masks(self):
        """
        Apply the currently defined masks to a set of localizations.

        """
        if not hasattr(self, "mask_edges_0"):
            print("Must define masks before applying to an image")
            return 

        # Prompt the user to select a file to apply the masks to 
        locs_file = getOpenFilePath(self.win, "Select localization CSV",
            "CSV files (*.csv)",
            initialdir=self.get_currdir())
        self.set_currdir(locs_file)
        locs = pd.read_csv(locs_file)

        # Only one set of masks defined - run static masking
        if not hasattr(self, "mask_edges_1"):
            mask_edges = [self.mask_edges_0[0]]
            mask_frames = [0]

        # Both sets of masks defined - run mask interpolation
        else:
            mask_edges = [self.mask_edges_0[0], self.mask_edges_1[0]]
            mask_frames = [0, int(locs["frame"].max()+1)]
        
        # Generate the mask interpolator
        print("Generating interpolator...")
        interpolator = QuotMaskInterpolator(mask_edges, mask_frames, n_vertices=101,
            interp_kind="linear", plot=False)

        # Plot the interpolated masks
        self.plot_mask_edges_interpolated(interpolator)

        # Run interpolation on the target set of localizations
        print("Running masking on the set of localizations...")
        locs["mask_index"] = interpolator(locs[['y', 'x']], locs['frame'],
            progress_bar=True)
        print(locs["mask_index"].unique())

        # Save to the same file
        print("Saving masked localizations to new files...")
        out_file_inside = "{}_in_mask.csv".format(os.path.splitext(locs_file)[0])
        out_file_outside = "{}_outside_mask.csv".format(os.path.splitext(locs_file)[0])
        locs[locs["mask_index"]>0].to_csv(out_file_inside, index=False)
        locs[locs["mask_index"]==0].to_csv(out_file_outside, index=False)
        print(locs["mask_index"].unique())

        print("Finished applying mask to localizations; now plotting...")

        point_sets = [
            interpolator.interpolate(frame)for frame \
                in np.linspace(
                    0,
                    int(locs["frame"].max()),
                    50
                ).astype(np.int64)
        ]

        show_mask_assignments(
            point_sets,
            locs,
            mask_col="mask_index",
            max_points_scatter=5000
        )

        print("Finished plotting")


    ## WIDGET CALLBACKS 

    def B_select_mask_0_callback(self):
        self.select_mask_file(0)

    def B_load_mask_0_callback(self):
        self.load_mask(0, plot=True, trial_interpolate=True)

    def B_select_mask_1_callback(self):
        self.select_mask_file(1)

    def B_generate_masks_0_callback(self):
        self.generate_masks(0, plot=True, trial_interpolate=True)

    def B_load_mask_1_callback(self):
        self.load_mask(1, plot=True, trial_interpolate=True) 

    def B_generate_masks_1_callback(self):
        self.generate_masks(1, trial_interpolate=True, plot=True)

    def B_apply_callback(self):
        self.apply_masks()

def show_mask_assignments(point_sets, locs, mask_col="mask_index",
    max_points_scatter=5000):
    """
    Show the assignment of a set of localizations to a set of masks. This
    is a three-panel plot that shows:

        (1) the mask definitions
        (2) the localization density
        (3) a scatter plot of localizations colored by mask membership

    args
    ----
        point_sets      :   list of 2D ndarray of shape (n_points, 2), the 
                            set of vertices defining each mask
        locs            :   pandas.DataFrame, the localizations
        mask_col        :   str, the column in the dataframe with the mask
                            assignments
        max_points_scatter: int, the maximum number of localizations to 
                            plot in the scatter plot (for memory efficiency)

    """
    # Only consider points that do not have the error flag set
    locs = locs[locs["error_flag"] == 0.0].copy()

    # Estimate the size of the ROI
    y_max = int(np.ceil(locs["y"].max()))
    x_max = int(np.ceil(locs["x"].max()))
    print(y_max, x_max)

    # Generate localization density
    y_bins = np.arange(y_max+1)
    x_bins = np.arange(x_max+1)
    H, _, _ = np.histogram2d(locs['x'], locs['y'], bins=(x_bins, y_bins))
    H = ndi.gaussian_filter(H, 5.0)

    # The set of points to use for the scatter plot
    if len(locs) > max_points_scatter:
        print("Only plotting %d/%d localizations..." % (max_points_scatter, len(locs)))
        L = np.asarray(locs[:max_points_scatter][["y", "x", mask_col]])
    else:
        L = np.asarray(locs[["x", "y", mask_col]])

    # Categorize each localization as either (1) assigned or (2) not assigned
    # to a mask
    inside = L[:,2] > 0
    outside = ~inside 

    # Localization density in the vicinity of each spot
    yx_int = L[:,:2].astype(np.int64)
    densities = H[yx_int[:,0], yx_int[:,1]]
    norm = Normalize(vmin=0, vmax=densities.max())

    # Make the 3-panel plot
    plt.close('all')
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))

    colors = sns.color_palette("magma", len(point_sets))
    for i, point_set in enumerate(point_sets):
        ax[0].plot(point_set[:,1], point_set[:,0], color=colors[i])
    ax[0].set_xlim((0, x_max))
    ax[0].set_ylim((0, y_max))
    ax[0].set_aspect("equal")

    ax[1].imshow(H.T, cmap='gray', origin="lower")
    ax[2].scatter(
        L[inside, 0],
        L[inside, 1],
        c=densities[inside],
        cmap="viridis",
        norm=norm,
        s=30
    )
    ax[2].scatter(
        L[outside, 0],
        L[outside, 1],
        cmap="magma",
        c=densities[outside],
        norm=norm,
        s=30
    )
    ax[2].set_xlim((0, x_max))
    ax[2].set_ylim((0, y_max))
    ax[2].set_aspect('equal')

    # Subplot labels
    ax[0].set_title("Mask definitions")
    ax[1].set_title("Localization density")
    ax[2].set_title("Inside/outside")

    # Show to the user. Unfortunately, matplotlib doesn't play very
    # nice with pyqtgraph and we frequently get an AttributeError here
    # when pyqtgraph tries to fuck with matplotlib's colors.
    try:
        plt.show()
    except AttributeError:
        pass 