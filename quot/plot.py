#!/usr/bin/env python
"""
plot.py -- simple plotting utilities for quot

"""
import sys
import os 

# Numeric
import numpy as np 
from scipy import ndimage as ndi 

# Image file reader
from nd2reader import ND2Reader 

# Dataframes, for handling trajectories
import pandas as pd 

# Core matplotlib library
import matplotlib.pyplot as plt 

# Color schema
from matplotlib import cm
from matplotlib import colors as mpl_colors 

# 3D plots
from mpl_toolkits.mplot3d import Axes3D

########################
## PLOTTING UTILITIES ##
########################

def hex_cmap(cmap, n_colors):
    """
    Generate a matplotlib colormap as a list of hex colors
    indices.

    args
    ----
        cmap        :   str, name of a matplotlib cmap (for 
                        instance, "viridis")
        n_colors    :   int

    returns
    -------
        list of str, hex color codes

    """
    C = cm.get_cmap(cmap, n_colors)
    return [mpl_colors.rgb2hex(j[:3]) for j in C.colors]

def kill_ticks(axes, spines=True):
    """
    Remove the y and x ticks from a plot.

    args
    ----
        axes        :   matplotlib.axes.Axes
        spines      :   bool, also remove the spines

    returns
    -------
        None

    """
    axes.set_xticks([])
    axes.set_yticks([])
    if spines:
        for s in ['top', 'bottom', 'left', 'right']:
            axes.spines[s].set_visible(False)

def wrapup(out_png, dpi=600):
    """
    Save a figure to a PNG.

    args
    ----
        out_png         :   str, save filename
        dpi             :   int, resolution

    """
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()
    if sys.platform == "darwin":
        os.system("open {}".format(out_png))

####################
## IMAGE PLOTTING ##
####################

def imshow(*imgs, vmax_mod=1.0, plot=True):
    """
    Show a variable number of images side-by-side in a 
    temporary window.

    args
    ----
        imgs        :   2D ndarrays
        vmax_mod    :   float, modifier for the
                        white point as a fraction of 
                        max intensity

    """
    n = len(imgs)
    if n == 1:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(imgs[0], cmap='gray', vmax=vmax_mod*imgs[0].max())
    else:
        fig, ax = plt.subplots(1, n, figsize=(3*n, 3))
        for j in range(n):
            ax[j].imshow(imgs[j], cmap='gray', vmax=vmax_mod*imgs[j].max())

    if plot:
        plt.show(); plt.close()
    else:
        return fig, ax 

def max_int_proj(axes, nd2_path, vmax_perc=99, vmin=None, cmap="gray",
    pixel_size_um=0.16, scalebar=False):
    """
    Make a maximum intensity projection of a temporal image sequence
    in a Nikon ND2 file.

    args
    ---
        axes        :   matplotlib.axes.Axes
        nd2_path    :   str, path to a target ND2 file
        vmax_perc   :   float, percentile of the intensity histogram
                        to use for the upper contrast bound
        vmin        :   float
        cmap        :   str
        pixel_size_um:  float, size of pixels in um
        scalebar    :   bool, make a scalebar

    returns
    -------
        None; plots directly to *axes*

    """
    reader = ND2Reader(nd2_path)
    mip = reader.get_frame(0).astype(np.float64)
    for frame_idx in range(1, reader.metadata["total_images_per_channel"]):
        try:
            mip = np.maximum(mip, reader.get_frame(frame_idx))
        except:
            continue
    if vmin is None:
        vmin = mip.min()
    axes.imshow(mip, cmap=cmap, vmin=vmin,
        vmax=np.percentile(mip, vmax_perc))
    kill_ticks(axes)
    if scalebar:
        try:
            from matplotlib_scalebar.scalebar import ScaleBar
            sb = ScaleBar(pixel_size_um, "um", frameon=False, 
                color="w", location="lower left")
            axes.add_artist(sb)
        except ModuleNotFoundError:
            print("quot.plot.max_int_proj: must have matplotlib_scalebar " \
                "installed in order to use scalebars")

def imlocdensity(axes, tracks, ymax=None, xmax=None, bin_size=0.1,
    kernel=3.0, vmax_perc=99, cmap="gray", pixel_size_um=0.16,
    scalebar=False):
    """
    Given a set of localizations from a single FOV, plot localization
    density.

    args
    ----
        axes        :   matplotlib.axes.Axes
        tracks      :   pandas.DataFrame
        ymax        :   int, the height of the FOV in pixels
        xmax        :   int, the width of the FOV in pixels
        bin_size    :   float, the size of the histogram bins in 
                        terms of pixels
        kernel      :   float, the size of the Gaussian kernel 
                        used for density estimation
        vmax_perc   :   float, the percentile of the density  
                        histogram used to define the upper contrast
                        threshold
        cmap        :   str
        pixel_size_um   :   float, size of pixels in um
        scalebar    :   bool, make a scalebar. Requires the 
                        matplotlib_scalebar package.

    returns
    -------
        None; plots directly to *axes*

    """
    # If ymax and xmax are not provided, set them to the maximum
    # y and x coordinates observed in the data
    if ymax is None:
        ymax = int(tracks['y'].max()) + 1
    if xmax is None:
        xmax = int(tracks['x'].max()) + 1

    # Determine histogram binning scheme
    ybins = np.arange(0, ymax, bin_size)
    xbins = np.arange(0, xmax, bin_size)

    # Accumulate localizations into a histogram
    density = np.histogram2d(tracks['y'], tracks['x'],
        bins=(ybins, xbins))[0].astype(np.float64)

    # KDE
    density = ndi.gaussian_filter(density, kernel)

    # Plot the result
    if density.sum() == 0:
        s = axes.imshow(density, vmin=0, vmax=1, cmap=cmap)
    else:
        s = axes.imshow(density, vmin=0,
            vmax=np.percentile(density, vmax_perc), 
            cmap=cmap)

    # Kill ticks
    kill_ticks(axes)

    # Scalebar, if desired
    if scalebar:
        try:
            from matplotlib_scalebar.scalebar import ScaleBar
            sb = ScaleBar(pixel_size_um*bin_size, "um", 
                frameon=False, color="w", location="lower left")
            axes.add_artist(sb)
        except ModuleNotFoundError:
            print("WARNING: module matplotlib_scalebar must be installed " \
                "in order to use scalebars")

###############################
## PSF PLOTTING and 3D PLOTS ##
###############################

def wireframe_overlay(img, model, plot=True):
    """
    Make a overlay of two 2-dimensional functions.

    args
    ----
        img, model      :   2D ndarray (YX), with the
                            same shape 

    """
    assert img.shape == model.shape 
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(projection='3d')
    Y, X = np.indices(img.shape)
    ax.plot_wireframe(X, Y, img, color='k')
    ax.plot_wireframe(X, Y, model, color='r')

    if plot:
        plt.show(); plt.close()
    else:
        return fig, ax 

################################
## TRAJECTORY ATTRIBUTE PLOTS ##
################################

def locs_per_frame(axes, tracks, n_frames=None, kernel=5,
    fontsize=10, title=None):
    """
    Plot the number of localizations per frame.

    args
    ----
        axes        :   matplotlib.axes.Axes
        tracks      :   pandas.DataFrame
        n_frames    :   int, the number of frames in the
                        movie. If *None*, defaults to the 
                        maximum frame in *tracks*
        kernel      :   float, size of uniform kernel used
                        for smoothing

    returns
    -------
        None

    """
    if n_frames is None:
        n_frames = int(tracks["frame"].max()) + 1
    frame_indices = np.arange(0, n_frames+1)
    H = np.histogram(
        tracks["frame"],
        bins=frame_indices,
    )[0].astype(np.float64)
    if kernel > 0:
        H = ndi.uniform_filter(H, kernel)
    axes.plot(frame_indices[:-1], H, color="k")
    axes.set_xlabel("Frame", fontsize=fontsize)
    axes.set_ylabel("Spots per frame", fontsize=fontsize)
    if not title is None:
        axes.set_title(title, fontsize=fontsize)

############################
## JUMP LENGTH HISTOGRAMS ##
############################

def plotRadialDisps(radial_disps, bin_edges, frame_interval=0.00548, plot=True):
    """
    Plot a set of radial displacement histograms as a simple line plot.

    args
    ----
        radial_disps        :   2D ndarray of shape (n_intervals, n_bins),
                                a set of radial displacement histograms 
                                for potentiall several frame intervals binned
                                spatially
        bin_edges           :   1D ndarray of shape (n_bins+1), the bin edge
                                definitions 
        frame_interval      :   float, seconds between frames
        plot                :   bool, immediately make a temporary plot to 
                                show to the user. Otherwise return the
                                Figure and Axes objects used to generate 
                                the image.

    returns
    -------
        if plot: 
            None

        else:
            (
                matplotlib.figure.Figure,
                1D array of matplotlib.axes.Axes
            )

    """
    if len(radial_disps.shape) == 1:
        radial_disps = np.asarray([radial_disps])
    assert radial_disps.shape[1] == (bin_edges.shape[0]-1)

    # Get the shape of the histograms
    n_intervals, n_bins = radial_disps.shape 

    # Get the centers of each bin
    bin_size = bin_edges[1] - bin_edges[0]
    bin_c = bin_edges[:-1] + bin_size/2.0

    # Set up the plot
    fig, ax = plt.subplots(n_intervals, 1, figsize=(4, 1.5*n_intervals),
        sharex=True)
    if n_intervals == 1:
        ax = np.asarray([ax])

    colors = hex_cmap("viridis", n_intervals*2)

    for interval in range(n_intervals):

        # Normalize
        pmf = radial_disps[interval,:] / radial_disps[interval,:].sum()

        # Plot
        ax[interval].plot(bin_c, pmf*bin_size, color=colors[interval],
            linestyle='-', linewidth=1.5,
            label="%.1f ms" % (frame_interval*1000*(interval+1)))
        ax[interval].legend(frameon=False, loc='upper right')

    # Axis labels
    ax[-1].set_xlabel("Radial displacement ($\mu$m)")
    for j in range(len(ax)):
        ax[j].set_ylabel("PMF")

    if plot:
        plt.show(); plt.close()
    else:
        return fig, ax 

def plotRadialDispsBar(radial_disps, bin_edges, frame_interval=0.00548,
    model=None, plot=True):
    """
    Plot a set of radial displacements as a bar graph. Also
    overlay a model as a line plot if desired.

    args
    ----
        radial_disps            :   2D ndarray of shape (n_intervals,
                                    n_bins), the radial displacements
        bin_edges               :   1D ndarray of shape (n_bins+1), bin
                                    edge definitions
        frame_interval          :   float, in sec
        model                   :   2D ndarray of shape (n_intervals,
                                    n_bins), model at each point
        plot                    :   bool, show immediately

    returns
    -------
        if plot:
            None

        else:
            (
                matplotlib.figure.Figure,
                array of matlotlib.axes.Axes
            )

    """
    # Check user inputs
    if len(radial_disps.shape) == 1:
        radial_disps = np.asarray([radial_disps])
    assert radial_disps.shape[1] == (bin_edges.shape[0]-1)

    # Check model, if using
    if not model is None:
        assert model.shape == radial_disps.shape

    # Get the shape of the histograms
    n_intervals, n_bins = radial_disps.shape 

    # Get the centers of each bin
    bin_size = bin_edges[1] - bin_edges[0]
    bin_c = bin_edges[:-1] + 0.5*bin_size 

    # Set up the plot
    fig, ax = plt.subplots(n_intervals, 1, figsize=(4, 1.5*n_intervals),
        sharex=True)
    if n_intervals == 1:
        ax = np.asarray([ax])

    colors = hex_cmap("viridis", n_intervals*2)

    for interval in range(n_intervals):

        # Normalize
        pmf = bin_size * radial_disps[interval,:] / \
            radial_disps[interval,:].sum()

        # Plot
        ax[interval].bar(bin_c, pmf, color=colors[interval],
            linestyle='-', linewidth=1, edgecolor='k',
            width=bin_size*0.8,
            label="%.1f ms" % (frame_interval*1000*(interval+1)))

        # Model definition, if using
        if not model is None:
            ax[interval].plot(bin_c, model[interval,:], color='k',
                linestyle='--', linewidth=1.5, label="Model")

        # Legend: show the frame interval 
        ax[interval].legend(frameon=False, loc='upper right')

    # Axis labels
    ax[-1].set_xlabel("Radial displacement ($\mu$m)")
    for j in range(len(ax)):
        ax[j].set_ylabel("PMF")

    if plot:
        plt.show(); plt.close()
    else:
        return fig, ax

###################
## MISCELLANEOUS ##
###################

def plot_pixel_mean_variance(means, variances, origin_files=None,
    model_gain=None, model_bg=None):
    """
    Plot pixel mean vs. variance for one or several movies,
    overlaying a linear gain model on top if desired.

    Best called through quot.read.measure_camera_gain.

    args
    ----
        means           :   list of 1D ndarray, the pixel 
                            means for each movie to plot
        variances       :   list of 1D ndarray, the pixel
                            variances for each movie to plot
        origin_files    :   list of str, the labels for each
                            element in *means* and *variances*
        model_gain      :   float, the camera gain 
        model_bg        :   float, the camera BG

    """
    # Check user inputs
    assert len(means) == len(variances)
    if not origin_files is None:
        assert len(origin_files) == len(means)

    # Plot setup
    fig, ax = plt.subplots(figsize=(3, 3))
    cmap = cm.get_cmap("gray", len(means)+2)
    colors = [mpl_colors.rgb2hex(cmap(i)) for i in range(len(means)+2)]

    # Plot each set of means and variances in a different color
    for i, (pixel_means, pixel_vars) in enumerate(zip(means, variances)):
        if not origin_files is None:
            label = origin_files[i]
        else:
            label = None 
        ax.scatter(pixel_means, pixel_vars, s=20, color=colors[i+1], label=label)

    # Plot the model overlay
    if (not model_gain is None) and (not model_bg is None):
        mean_min = min([j.min() for j in means])
        mean_max = max([j.max() for j in means])
        model_means = np.linspace(mean_min, mean_max, 101)
        model_vars = (model_means - model_bg) * model_gain 
        ax.plot(model_means, model_vars, linestyle='--', color='k', label="Model")

    # Limit the y extent if there are variance outliers (which is common)
    var_upper = np.percentile(np.concatenate(variances), 99.9)
    ax.set_ylim((0, var_upper*1.5))

    # Labels
    ax.legend(frameon=False, prop={'size': 6}, loc="upper left")
    ax.set_xlabel("Pixel mean (AU)")
    ax.set_ylabel("Pixel variance (AU$^{2}$)")

    plt.tight_layout(); plt.show(); plt.close()

###################################
## MULTI-FILE PLOTTING FUNCTIONS ##
###################################

def imlocdensities(*csv_files, out_png=None, filenames=False, **kwargs):
    """
    For each of a set of CSV files, make a KDE for localization
    density. This is a wrapper around imlocdensity().

    If *out_png* is passed, this function saves the result to a file.
    Otherwise it plots to the screen.

    args
    ----
        csv_files       :   variadic str, a set of CSV files
        out_png         :   str, save filename
        filenames       :   str, include the filenames as the
                            plot title
        kwargs          :   keyword arguments to imlocdensity().
                            See that function's docstring for
                            more info. These include: ymax, xmax,
                            bin_size, kernel, vmax_perc, cmap,
                            pixel_size_um, scalebar

    """
    n = len(csv_files)
    if n == 0:
        print("quot.plot.imlocdensities: no files passed")
        return 

    # Number of subplots on an edge
    m = int(np.ceil(np.sqrt(n)))

    mx = m 
    my = n // m + 1

    # Lay out the main plot
    fig, ax = plt.subplots(my, mx, figsize=(3*mx, 3*my))

    # Make localization density plots for each file
    for i, csv_file in enumerate(csv_files):
        tracks = pd.read_csv(csv_file)
        imlocdensity(ax[i//mx, i%mx], tracks, **kwargs)

        # Set filename as plot title if desired
        if filenames:
            ax[i//mx, i%mx].set_title(csv_file, fontsize=8)

    # Set the remainder of the plots to invisible
    for i in range(n, my*mx):
        kill_ticks(ax[i//mx, i%mx])

    # Save or show
    if not out_png is None:
        wrapup(out_png)
    else:
        plt.tight_layout(); plt.show(); plt.close()

def locs_per_frame_files(*csv_files, out_png=None, **kwargs):
    """
    Given a set of trajectory CSVs, make a plot where each subpanel
    shows the number of localizations in that file as a function of
    time. 

    args
    ----
        csv_files       :   variadic str, paths to CSVs
        out_png         :   str, save file
        kwargs          :   to locs_per_frame()

    """
    n = len(csv_files)
    if n == 0:
        print("quot.plot.locs_per_frame_files: no files passed")
        return 

    # Number of subplots on an edge
    m = int(np.ceil(np.sqrt(n)))

    mx = m 
    my = n//m + 1

    # Make the plot
    fig, ax = plt.subplots(my, mx, figsize=(mx*3, my*1.5))
    for i, csv_file in enumerate(csv_files):
        tracks = pd.read_csv(csv_file)
        locs_per_frame(ax[i//mx, i%mx], tracks, title=csv_file, **kwargs)

    # Set unused subplots to blank
    for i in range(n, my*mx):
        kill_ticks(ax[i//mx, i%mx])

    # Save or show
    if not out_png is None:
        wrapup(out_png)
    else:
        plt.tight_layout(); plt.show(); plt.close()






