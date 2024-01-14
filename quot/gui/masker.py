#!/usr/bin/env python
"""
masker.py -- 

"""
import sys 

# File paths
import os 

# Numeric
import numpy as np 

# Uniform filter
from scipy import ndimage as ndi 

# TIF file reader
import tifffile

# Dataframes
import pandas as pd 

# Plotting
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize 

# Path object, for fast determination whether points lie
# inside a closed path
from matplotlib.path import Path 

# File readers
from ..read import ImageReader 

# Get the edges of a binary mask
from ..helper import (
    get_edges,
    get_ordered_mask_points
)

# Read *Tracked.mat format into a DataFrame
from ..helper import tracked_mat_to_csv

# Core GUI utilities
import PySide6
from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, \
    QVBoxLayout, QGridLayout, QDialog 

# pyqtgraph plotting utilities
from pyqtgraph import ImageView, ScatterPlotItem, PolyLineROI

# Custom GUI utilities
from .guiUtils import (
    IntSlider,
    getSaveFilePath,
    getOpenFilePath,
    getTextInputs,
    SingleComboBoxDialog
)

class Masker(QDialog):
    """
    An interface for the user to draw masks on an image or movie.
    This includes:

        - draw masks on the raw image, either by placing discrete vertices
            or freestyle drawing
        - modify masks by adding / deleting / moving vertices
        - change between frames
        - export masks
        - apply masks to image files

    init
    ----
        image_path           :   str, path to an image file (e.g. ND2 or TIF)
        max_points_freestyle :  int, the maximum number of points to allow
                                for a freestyle mask. This limits memory
                                usage.
        parent               :  root QWidget 


    """
    def __init__(self, image_path, max_points_freestyle=40, dialog_mode=False, parent=None):

        super(Masker, self).__init__(parent=parent)
        self.image_path = image_path 
        self.max_points_freestyle = max_points_freestyle
        self.dialog_mode = dialog_mode 
        self.initData()
        self.initUI()

    def initData(self):
        """
        Load the images and related data.

        """
        # Create the image reader
        self.imageReader = ImageReader(self.image_path)

        # Get the first image
        self.image = self.imageReader.get_frame(0)

    def initUI(self):
        """
        Initialize the user interface.

        """
        # Main layout
        L_master = QGridLayout(self)

        # An ImageView on the left to contain the subject of masking
        self.imageView = ImageView(parent=self)
        L_master.addWidget(self.imageView, 0, 0, 15, 2)
        self.imageView.setImage(self.image)

        # Override the default ImageView.imageItem.mouseClickEvent
        self.imageView.imageItem.mouseClickEvent = self._imageView_mouseClickEvent

        # Override the default ImageView.imageItem.mouseDoubleClickEvent 
        self.imageView.imageItem.mouseDoubleClickEvent = \
            self._imageView_mouseDoubleClickEvent 

        # All currently clicked points
        self.points = []

        # All current PolyLineROI objects
        self.polyLineROIs = []

        # ScatterPlotItem, to show current accumulated clicks
        self.scatterPlotItem = ScatterPlotItem()
        self.scatterPlotItem.setParentItem(self.imageView.imageItem)

        ## WIDGETS
        widget_align = Qt.AlignTop

        # Frame slider
        self.frame_slider = IntSlider(
            minimum=0,
            maximum=self.imageReader.n_frames-1,
            interval=1,
            init_value=0,
            name='Frame',
            parent=self
        )
        L_master.addWidget(self.frame_slider, 0, 2, 1, 1,
            alignment=widget_align)
        self.frame_slider.assign_callback(self.frame_slider_callback)

        # Button: create new ROI
        self.create_roi_mode = False 
        self.B_create_ROI = QPushButton("Draw ROI", parent=self)
        self.B_create_ROI.clicked.connect(self.B_create_ROI_callback)
        L_master.addWidget(self.B_create_ROI, 10, 2, 1, 1, 
            alignment=widget_align)

        # Button: freestyle drawing to make ROI
        self.freestyle_mode = False 
        self.B_freestyle = QPushButton("Freestyle", parent=self)
        self.B_freestyle.clicked.connect(self.B_freestyle_callback)
        L_master.addWidget(self.B_freestyle, 11, 2, 1, 1, 
            alignment=widget_align)

        # Pure dialog mode: no options to save or load masks, only 
        # define and accept
        if self.dialog_mode:

            # Button: accept the current set of masks. This is only meaningful
            # when this class is being used as a dialog by other classes.
            self.B_accept = QPushButton("Accept masks", parent=self)
            self.B_accept.clicked.connect(self.B_accept_callback)
            L_master.addWidget(self.B_accept, 12, 2, 1, 1, alignment=widget_align)

        # Non-dialog mode: full range of options
        else:

            # Button: apply these masks to the localizations in a file
            self.B_apply = QPushButton("Apply masks", parent=self)
            self.B_apply.clicked.connect(self.B_apply_callback)
            L_master.addWidget(self.B_apply, 12, 2, 1, 1, 
                alignment=widget_align)

            # Button: save masks to a file
            self.B_save = QPushButton("Save masks", parent=self)
            self.B_save.clicked.connect(self.B_save_callback)
            L_master.addWidget(self.B_save, 13, 2, 1, 1, 
                alignment=widget_align)

            # Button: load preexisting masks from a file
            self.B_load = QPushButton("Load masks", parent=self)
            self.B_load.clicked.connect(self.B_load_callback)
            L_master.addWidget(self.B_load, 14, 2, 1, 1, 
                alignment=widget_align)

        # Resize and launch
        self.resize(800, 600)
        self.show()

    ## CORE FUNCTIONS

    def update_image(self, frame_index=None, autoRange=False, 
        autoLevels=False, autoHistogramRange=False):
        if frame_index is None:
            frame_index = self.frame_slider.value()
        self.image = self.imageReader.get_frame(frame_index)
        self.imageView.setImage(self.image, autoRange=autoRange,
            autoLevels=autoLevels, autoHistogramRange=autoHistogramRange)

    def update_scatter(self):
        """
        Write the contents of self.points to self.scatterPlotItem.

        """
        self.scatterPlotItem.setData(
            pos=np.asarray(self.points),
            pxMode=False,
            symbol='o',
            pen={'color': '#FFFFFF', 'width': 3.0},
            size=2.0,
            brush=None,
        )

    def clear_scatter(self):
        self.points = []
        self.scatterPlotItem.setData()

    def createPolyLineROI(self, points):
        p = PolyLineROI(self.points, closed=True, removable=True)
        self.polyLineROIs.append(p)
        self.imageView.view.addItem(self.polyLineROIs[-1])

        # If the user requests to remove this ROI, remove it
        self.polyLineROIs[-1].sigRemoveRequested.connect(self._remove_ROI)

    def getPoints(self, polyLineROI):
        """
        Return the set of points that make up a PolyLineROI as a 
        2D ndarray (shape (n_points, 2)).

        """
        positions = polyLineROI.getSceneHandlePositions()
        return np.asarray(
            [[polyLineROI.mapSceneToParent(p[1]).x(), 
              polyLineROI.mapSceneToParent(p[1]).y()] for p in positions])

    def get_currdir(self):
        """
        Get the last directory that was accessed by the user. If 
        no directory was previously accessed, then this is just 
        the same directory as the image file.

        """
        if not hasattr(self, "_currdir"):
            self.set_currdir(self.image_path)
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

    def clear_polyLineROIs(self):
        """
        Destroy all current PolyLineROI objects.

        """
        for p in self.polyLineROIs[::-1]:
            self._remove_ROI(p)

    ## SIGNAL CALLBACKS

    def _imageView_mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            if self.imageView.imageItem.raiseContextMenu(ev):
                ev.accept()
        if ev.button() == QtCore.Qt.LeftButton and self.create_roi_mode:
            self.points.append(ev.pos())
            self.update_scatter()

    def _imageView_mouseDoubleClickEvent(self, ev):
        if self.create_roi_mode:
            self.createPolyLineROI(self.points)
            self.clear_scatter()
            self.create_roi_mode = False 
            self.imageView.view.state['mouseEnabled'] = np.array([True, True])
        elif self.freestyle_mode:
            self.B_freestyle_callback()
        else:
            pass 

    def _remove_ROI(self, polylineroi):
        polylineroi.sigRemoveRequested.disconnect(self._remove_ROI)
        self.polyLineROIs.remove(polylineroi)
        self.imageView.view.removeItem(polylineroi)
        del polylineroi 

    ## WIDGET CALLBACKS

    def frame_slider_callback(self):
        """
        Callback for user changes to the frame slider, which changes 
        the current frame shown beneath the masks.

        """
        frame_index = self.frame_slider.value()
        self.update_image(frame_index=frame_index)

    def B_create_ROI_callback(self):
        """
        Callback for user selection of the "Draw ROI" button. Enter
        create ROI mode, which allows the user to place discrete vertices
        sequentially to build a mask.

        """
        self.create_roi_mode = True
        self.imageView.view.state['mouseEnabled'] = np.array([False, False])

    def B_freestyle_callback(self):
        """
        Callback for user selection of the "freestyle" button. Enter
        freestyle mode, which allows the user to draw a mask on the 
        image by dragging the mouse.

        """
        # End freestyle mode by creating a mask from the drawn points
        if self.freestyle_mode:
            self.freestyle_mode = False 
            self.imageView.imageItem.setDrawKernel(kernel=None,
                mask=None, center=None)
            mask = (self.image == self.draw_val)
            self.points = get_ordered_mask_points(
                mask, 
                max_points=self.max_points_freestyle
            )
            self.createPolyLineROI(self.points)
            self.frame_slider_callback()
            self.points = []
            self.B_freestyle.setText("Freestyle")

        # Start freestyle mode by enabling drawing on self.imageView.imageItem
        else:
            self.freestyle_mode = True 
            self.draw_val = int(self.image.max() + 2)
            kernel = np.array([[self.draw_val]])
            self.imageView.imageItem.setDrawKernel(kernel=kernel, 
                mask=None, center=(0,0))
            self.B_freestyle.setText("Finish freestyle")

    def B_save_callback(self):
        """
        Save the current set of masks to a file. Every currently 
        defined mask is saved in a CSV-like format indexed by mask
        vertex. This CSV has the following columns:

            filename        :   str, the source filename
            mask_index      :   int, the index of this mask
            y               :   float, the y-coordinate of the vertex
            x               :   float, the x-coordinate of the vertex
            vertex          :   int, the index of this vertex in the
                                context of its mask

        The "vertex" always ascends from 0 to n-1, where n is the number
        of vertices in that mask.

        """
        # If no masks are defined, do nothing
        if len(self.polyLineROIs) == 0:
            return 

        # Prompt the user to select an output filename
        out_path = getSaveFilePath(self, "Select output CSV",
            "{}_masks.csv".format(os.path.splitext(self.image_path)[0]),
            "CSV files (*.csv)", initialdir=self.get_currdir())

        # For each PolyLineROI mask, get the corresponding set of vertices
        point_arrays = [self.getPoints(p) for p in self.polyLineROIs]
        n_masks = len(point_arrays)

        # Get the total size of the output dataframe
        m = sum([arr.shape[0] for arr in point_arrays])

        # Format as a pandas.DataFrame
        df = pd.DataFrame(index=np.arange(m), columns=["filename", "mask_index", "y", "x", "vertex"])
        df["filename"] = self.image_path 
        c = 0
        for mask_index, arr in enumerate(point_arrays):
            l = arr.shape[0]
            df.loc[c:c+l-1, "mask_index"] = mask_index 
            df.loc[c:c+l-1, "y"] = arr[:,0]
            df.loc[c:c+l-1, "x"] = arr[:,1]
            df.loc[c:c+l-1, "vertex"] = np.arange(l)
            c += l 

        # Save
        df.to_csv(out_path, index=False)

        # Save this directory as the last accessed
        self.set_currdir(out_path)

    def B_load_callback(self):
        """
        Load a set of masks from a file previously saved with this GUI. This
        erases any currently defined masks.

        """
        # Prompt the user to select a file
        path = getOpenFilePath(self, "Select mask CSV", "CSV and TIF files (*.csv *.tif *.tiff)",
            initialdir=self.get_currdir())
        self.set_currdir(path)

        ext = os.path.splitext(path)[-1]

        # Open this file and check that it contains the necessary info
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

            # Warn the user if the image file path in this file doesn't
            # match the current image file path 
            if df.loc[0, "filename"] != self.image_path:
                print("WARNING: original file path for this mask file is different " \
                    "than the current image file.\nOriginal: {}\nCurrent: {}".format(
                        df.loc[0, "filename"], self.image_path))
        elif ext in [".tif", ".tiff"]:

            # Load the mask image
            masks_im = tifffile.imread(path)
            assert len(masks_im.shape) == 2, "WARNING: RGB TIFs not supported"

            # Get the set of unique masks defined in this image
            n_points = 100
            mask_indices = [j for j in list(np.unique(masks_im)) if j != 0]
            dfs = []
            for mi in mask_indices:
                mask_im = masks_im == mi
                edges = get_edges(mask_im)
                points = get_ordered_mask_points(edges, max_points=n_points)
                df = pd.DataFrame(index=list(range(points.shape[0])), 
                    columns=["filename", "mask_index", "y", "x"])
                df["filename"]= path 
                df["mask_index"] = mi
                df["y"] = points[:,0]
                df["x"] = points[:,1]
                df["vertex"] = list(range(points.shape[0]))
                dfs.append(df)
            df = pd.concat(dfs, axis=0, ignore_index=True)

        # Erase the current set of masks, if any 
        self.clear_polyLineROIs()

        # Generate a PolyLineROI object for each of the loaded masks
        for mask_index, mask_frame in df.groupby("mask_index"):
            self.points = np.asarray(mask_frame[["y", "x"]])
            self.createPolyLineROI(self.points)
            self.points = []

    def B_apply_callback(self):
        """
        Callback for user selection of the "Apply masks" button. The
        user is prompted to select a file containing localizations,
        and the localizations in that file are classified according to 
        which mask they belong to.

        """
        # Prompt the user to select a file containing localizations
        path = getOpenFilePath(self, "Select localization file", 
            "CSV files and *Tracked.mat files (*.csv *Tracked.mat)",
            initialdir=self.get_currdir())
        self.set_currdir(path)

        # Open this file and make sure it actually contains localizations
        ext = os.path.splitext(path)[-1]
        try:
            if ext == ".csv":
                locs = pd.read_csv(path)
                assert all([c in locs.columns for c in ['frame', 'y', 'x']])
            elif ext == ".mat":
                locs = tracked_mat_to_csv(path)
        except:
            print("Could not load file {}; must either be *.csv or " \
                "*Tracked.mat format".format(path))
            return 

        # Prompt the user to input the mask column. This can be something 
        # as simple as "mask_index", but the user may want something more
        # descriptive - like "nuclear_mask" or something. This becomes useful
        # when the user applies masks multiple times to the same file - for
        # instance, to label primary and secondary features (e.g. nuclei and
        # nucleoli) in an image.
        col = getTextInputs(["Output mask column name"], ["mask_index"],
            title="Select output column name")[0]

        # Prompt the user to select a mode for the assignment of trajectories
        # to masks. This can either be "by_localization" (independent of the 
        # trajectory to which each localization is assigned), "single_point"
        # (assign all localizations to a mask if a single localization is 
        # inside the mask), or "all_points" (only assign all localizations to a 
        # mask if all localizations are inside the mask)
        options = ["by_localization", "single_point", "all_points"]
        box = SingleComboBoxDialog("Method by which trajectories are assigned to masks",
            options, init_value="single_point", title="Assignment mode", parent=self)
        box.exec_()
        if box.result() == 1:
            mode = box.return_val
        else:
            print("Dialog not accepted")
            return

        # Assign each localization to one of the current masks
        point_sets = [self.getPoints(p) for p in self.polyLineROIs]
        locs[col] = apply_masks(point_sets, locs, mode=mode)

        # Save to the same file
        if ext == ".csv":
            out_path = path 
        elif ext == ".mat":
            if "_Tracked.mat" in path:
                out_path = path.replace("_Tracked.mat", ".csv")
            elif "Tracked.mat" in path:
                out_path = path.replace("Tracked.mat", ".csv")
            else:
                out_path = "{}.csv".format(os.path.splitext(path)[0])
        locs.to_csv(out_path, index=False)

        # Save only the masked trajectories to a different file
        out_file_inside = "{}_in_mask.csv".format(os.path.splitext(out_path)[0])
        out_file_outside = "{}_outside_mask.csv".format(os.path.splitext(out_path)[0])
        locs[locs["mask_index"] > 0].to_csv(out_file_inside, index=False)
        locs[locs["mask_index"] == 0].to_csv(out_file_outside, index=False)

        # Show the result
        show_mask_assignments(point_sets, locs, mask_col=col, max_points_scatter=5000)

    def B_accept_callback(self):
        """
        If this GUI is being used as a dialog, accept and return the currently
        defined set of masks.

        """
        self.return_val = [self.getPoints(p) for p in self.polyLineROIs]
        print(self.return_val)
        self.accept()

def inside_mask(points, locs, mode="single_point"):
    """
    args
    ----
        points  :   2D ndarray of shape (n_points, 2), the Y 
                    and X points of each vertex defining the mask 
                    edge, ordered so that they are adjacent in the
                    mask.
        locs    :   pandas.DataFrame, the set of localizations
        mode    :   str, either "by_localization", "single_point"
                    or "all_points", the criterion by which 
                    localizations/trajectories are assigned to 
                    masks.

                    If "by_localization", then each localization
                    is assigned to the mask depending on whether
                    or not it is inside the mask, regardless of 
                    the other points in the trajectory.

                    If "single_point", then if a single point from
                    each trajectory is inside the mask, all points
                    from that trajectory are labeled as inside the
                    mask.

                    If "all_points", then a trajectory must have 
                    all of its points inside the mask in order to be
                    assigned to that mask. Otherwise, none of its
                    localizations are assigned to the mask.

    returns
    -------
        1D ndarray of shape (len(locs),), dtype bool;
            whether each localization lies inside the mask

    """
    path = Path(points, closed=True)
    locs["inside_mask"] = path.contains_points(locs[['y', 'x']])
    if mode == "single_point":
        locs = locs.join(
            locs.groupby("trajectory")["inside_mask"].any().rename("inside_mask_by_track"),
            on="trajectory"
        )
        locs["inside_mask"] = locs["inside_mask_by_track"]
        locs = locs.drop("inside_mask_by_track", axis=1)
    elif mode == "all_points":
        locs = locs.join(
            locs.groupby("trajectory")["inside_mask"].all().rename("inside_mask_by_track"),
            on="trajectory"
        )
        locs["inside_mask"] = locs["inside_mask_by_track"]
        locs = locs.drop("inside_mask_by_track", axis=1)       
    locs_inside = np.asarray(locs["inside_mask"]).astype(bool)
    locs = locs.drop("inside_mask", axis=1)
    return locs_inside 

def inside_mask_old(points, locs):
    """
    Given a set of points defining a mask, designate each localization as 
    either inside or outside of the mask.

    args
    ----
        points          :   2D ndarray of shape (n_points, 2), the Y 
                            and X points of each vertex defining the mask 
                            edge, ordered so that they are adjacent in the
                            mask.
        locs            :   pandas.DataFrame, the set of localizations

    returns
    -------
        1D ndarray of shape (len(locs),), dtype bool;
            whether each localization lies inside the mask

    """
    # Construct a matplotlib.path.Path object for fast determination of 
    # whether points lie inside or outside the mask
    path = Path(points, closed=True)

    # Assign each point to a mask
    return path.contains_points(locs[["y", "x"]])

def apply_masks(point_sets, locs, mode="single_point"):
    """
    Assign each of a set of localizations to one of several masks.

    The masks are assumed to be mutually exclusive - if a localization is 
    found to lie within multiple masks, then it is assigned to the last
    one.

    args
    ----
        point_sets      :   list of 2D ndarray of shape (n_points, 2), the 
                            set of vertices defining each mask
        locs            :   pandas.DataFrame, the localizations
        mode            :   str, either "by_localization", "single_point",
                            or "all_points", the mode by which trajectories
                            are assigned to masks

    returns
    -------
        1D ndarray of shape (len(locs),), dtype int64, the index of the
            mask to which each localization belongs. If 0, then the localization
            did not fall into any mask.

    """
    locs_copy = locs.copy()
    assigned = np.zeros(len(locs_copy), dtype=np.int64)
    for i, point_set in enumerate(point_sets):
        assigned[inside_mask(point_set, locs_copy, mode=mode)] = i+1 
    return assigned 

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

    # Generate coordinates for each pixel
    Y, X = np.indices((y_max, x_max))
    YX = np.asarray([Y.ravel(), X.ravel()]).T 

    # Generate an image where each pixel is assigned to a mask
    mask_im = np.zeros((y_max, x_max), dtype=np.int64)
    for i, point_set in enumerate(point_sets):
        path = Path(point_set, closed=True)
        mask_im[path.contains_points(YX).reshape((y_max, x_max))] = i+1

    # Generate localization density
    y_bins = np.arange(y_max+1)
    x_bins = np.arange(x_max+1)
    H, _, _ = np.histogram2d(locs['y'], locs['x'], bins=(y_bins, x_bins))
    H = ndi.gaussian_filter(H, 5.0)

    # The set of points to use for the scatter plot
    if len(locs) > max_points_scatter:
        print("Only plotting %d/%d localizations..." % (max_points_scatter, len(locs)))
        L = np.asarray(locs[:max_points_scatter][["y", "x", mask_col]])
    else:
        L = np.asarray(locs[["y", "x", mask_col]])

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

    ax[0].imshow(mask_im, cmap='gray')
    ax[1].imshow(H, cmap='gray')
    ax[2].scatter(
        L[inside, 1],
        y_max-L[inside, 0],
        c=densities[inside],
        cmap="viridis",
        norm=norm,
        s=30
    )
    ax[2].scatter(
        L[outside, 1],
        y_max-L[outside, 0],
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

def reconstruct_mask(mask_csv, shape):
    """
    Given a mask CSV of the type produced by B_save_callback in Masker,
    produce a binary mask on some pixel grid.

    args
    ----
        mask_csv        :   str, path to CSV
        shape           :   2-tuple of int, the shape of the output image

    returns
    -------
        2D ndarray of dtype bool, the binary masks

    """
    mask_df = pd.read_csv(mask_csv)
    mask_indices = mask_df["mask_index"].unique()
    result = np.zeros(shape, dtype=bool)
    Y, X = np.indices(shape)
    YX = np.zeros((shape[0] * shape[1], 2))
    YX[:,0] = Y.ravel()
    YX[:,1] = X.ravel()
    for mask_index in mask_indices:
        mask = mask_df.loc[mask_df["mask_index"] == mask_index]
        p = Path(np.asarray(mask[['y', 'x']]), closed=True)
        result = np.logical_or(result, p.contains_points(YX).reshape(shape))
    return result 
