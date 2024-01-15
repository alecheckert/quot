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

def plot_jump_pdfs(axes, tracks, n_frames=4, pixel_size_um=0.16, bin_size=0.02,
    norm=True, frame_interval=0.00748, fontsize=10, cmap="viridis",
    start_frame=0, max_jumps_per_track=4, use_entire_track=False, **kwargs):
    """
    Plot the empirical jump length probability density function.

    args
    ----
        axes            :   matplotlib.axes.Axes
        tracks          :   pandas.DataFrame
        n_frames        :   int, the maximum number of frame 
                            intervals to consider
        pixel_size_um   :   float, size of pixels in um
        bin_size        :   float, size of the bins to use in 
                            the plotted histogram
        norm            :   bool, normalize to a PDF
        frame_interval  :   float, frame interval in seconds
        fontsize        :   int
        cmap            :   str
        start_frame     :   int, disregard jumps before this frame
        max_jumps_per_track     :   int, the maximum number of jumps
                                    to consider from any one track
        use_entire_track:   bool, use all jumps from every track
        kwargs          :   to rad_disp_histogram_2d

    """
    if start_frame > 0:
        tracks = tracks[tracks["frame"] >= start_frame]

    # Make the complete jump length histogram
    H, bin_edges = rad_disp_histogram_2d(tracks, n_frames=n_frames,
        pixel_size_um=pixel_size_um, use_entire_track=use_entire_track,
        max_jumps_per_track=max_jumps_per_track, **kwargs)

    # Aggregate histogram bins
    factor = int(bin_size // 0.001)
    H, bin_edges = coarsen_histogram(H, bin_edges, factor)
    bin_size = bin_edges[1] - bin_edges[0]
    bin_c = bin_edges[:-1] + bin_size * 0.5

    if norm:
        H = H.astype(np.float64)
        for frame_idx in range(n_frames):
            if H[frame_idx, :].sum() > 0:
                H[frame_idx, :] = H[frame_idx, :] / H[frame_idx, :].sum()

    # Plot
    colors = hex_cmap(cmap, n_frames+2)
    for frame_idx in range(n_frames):
        axes.plot(bin_c, H[frame_idx, :], color=colors[frame_idx],
            label="$\Delta t = $ %.4f sec" % (frame_interval * (frame_idx+1)))
    axes.set_xlabel("2D radial jump ($\mu$m)", fontsize=fontsize)
    axes.set_ylabel("Frequency")
    axes.legend(frameon=False, loc="upper right", prop={'size': 8})

def plot_jump_cdfs(axes, tracks, n_frames=4, pixel_size_um=0.16,
    frame_interval=0.00748, fontsize=10, cmap="viridis", 
    start_frame=0, max_jumps_per_track=4, use_entire_track=False, 
    linewidth=1, plot_max_r=None, **kwargs):
    """
    Plot the empirical jump length probability cumulative
    distribution function.

    args
    ----
        axes            :   matplotlib.axes.Axes
        tracks          :   pandas.DataFrame
        n_frames        :   int, the maximum number of frame 
                            intervals to consider
        pixel_size_um   :   float, size of pixels in um
        bin_size        :   float, size of the bins to use in 
                            the plotted histogram
        frame_interval  :   float, frame interval in seconds
        fontsize        :   int
        cmap            :   str
        start_frame     :   int, disregard jumps before this frame
        max_jumps_per_track     :   int, the maximum number of jumps
                                    to consider from any one track
        use_entire_track:   bool, use all jumps from every track
        linewidth       :   int, width of the lines
        plot_max_r      :   float, the maximum jump length to show
                            in the plot
        kwargs          :   to rad_disp_histogram_2d

    """
    if start_frame > 0:
        tracks = tracks[tracks['frame'] >= start_frame]
    H, bin_edges = rad_disp_histogram_2d(tracks, n_frames=n_frames,
        pixel_size_um=pixel_size_um, max_jumps_per_track=max_jumps_per_track,
        use_entire_track=use_entire_track, **kwargs)
    H = H.astype(np.float64)
    cdfs = np.cumsum(H, axis=1)
    for i in range(n_frames):
        cdfs[i,:] = cdfs[i,:] / cdfs[i,-1]

    colors = hex_cmap(cmap, n_frames+2)
    for i in range(n_frames):
        axes.plot(bin_edges[1:], cdfs[i,:], color=colors[i],
            linewidth=linewidth,
            label="$\Delta t = $ %.4f sec" % (frame_interval * (i+1)))

    axes.set_xlabel("2D radial jump ($\mu$m)", fontsize=fontsize)
    axes.set_ylabel("CDF")
    axes.legend(frameon=False, loc="lower right", prop={'size': 8})
    if not plot_max_r is None:
        axes.set_xlim((axes.get_xlim()[0], plot_max_r))

# rad_disp_histogram_2d(tracks, n_frames=4, bin_size=0.001, 
#     max_jump=5.0, pixel_size_um=0.160, n_gaps=0, use_entire_track=False,
#     max_jumps_per_track=10)

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

##########################
## ANGULAR DISTRIBUTION ##
##########################

def angular_dist(axes, tracks, min_disp=0.2, delta=1, n_bins=50, norm=True,
    pixel_size_um=0.16, bottom=0.02, angle_ticks=True):
    """
    Plot the angular distribution of displacements for a 
    set of trajectories.

    Note that *axes* must be generated with projection = "polar".
    For instance, 

        fig = plt.figure()
        axes = fig.add_subplot(projection="polar")

    args
    ----
        axes        :   matplotlib.axes.Axes
        tracks      :   pandas.DataFrame
        min_disp    :   float, the minimum displacement required
                        to consider a displacement for an angle
                        in um
        delta       :   int, the number of subsequent jumps over
                        which to calculate the angle. For example,
                        if *delta* is 1, then angles are calculated
                        between subsequent jumps in a trajectory.
                        If *delta* is 2, then angles are calculated
                        between jump n and jump n+2, etc.
        n_bins      :   int, number of histogram bins
        norm        :   bool, normalize histogram
        pixel_size_um   :   float, size of pixels in um
        angle_ticks :   bool, do tick labels for the angles

    returns
    -------
        None, plots directly to *axes*

    """
    # Calculate angles
    angles = bond_angles(tracks, min_disp=min_disp/pixel_size_um, delta=delta)

    # Make a histogram
    bin_edges = np.linspace(0, np.pi, n_bins+1)
    bin_size = bin_edges[1] - bin_edges[0]
    bin_c = bin_edges[:-1] + bin_size * 0.5
    H = np.histogram(angles, bins=bin_edges)[0]

    if norm and H.sum() > 0:
        H = H.astype(np.float64) / H.sum()

    # Plot
    bars = axes.bar(bin_c, H, color="w", edgecolor="k",
        width=bin_size, bottom=bottom)
    axes.plot(bin_c, [bottom for i in range(n_bins)],
        color="k", linewidth=1) 
    yt = np.asarray(axes.get_yticks())
    axes.set_yticks(yt[yt>=bottom])
    axes.set_yticklabels([])
    axes.set_thetamin(0)
    axes.set_thetamax(180)
    axes.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi])
    if not angle_ticks:
        axes.set_xticklabels([])
    axes.set_ylim((0, 1.1*axes.get_ylim()[1]))



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

def jump_cdf_files(*csv_files, out_png=None, **kwargs):
    """
    For each of a set of trajectory CSVs, plot the jump
    length CDFs.

    args
    ----
        csv_files       :   list of str, paths to CSV files
        out_png         :   str, output plot file
        kwargs          :   to plot_jump_cdfs(). These include:
                            n_frames, pixel_size_um, frame_interval,
                            fontsize, cmap, start_frame,
                            max_jumps_per_track, use_entire_track,
                            linewidth

    returns
    -------
        None; either plots to screen or saves depending
            on whether *out_png* is set

    """
    n = len(csv_files)
    if n == 0:
        print("quot.plot.jump_cdfs_files: no files passed")
        return 

    m = int(np.ceil(np.sqrt(n)))
    mx = m 
    my = n//m + 1

    fig, ax = plt.subplots(my, mx, figsize=(4*mx, 2*my))
    for i, csv_file in enumerate(csv_files):
        tracks = pd.read_csv(csv_file)
        plot_jump_cdfs(ax[i//mx, i%mx], tracks, **kwargs)
        ax[i//mx, i%mx].set_title(csv_file, fontsize=10)

    for i in range(n, my*mx):
        kill_ticks(ax[i//mx, i%mx])

    if not out_png is None:
        wrapup(out_png)
    else:
        plt.tight_layout(); plt.show(); plt.close()

def angular_dist_files(*csv_files, out_png=None, start_frame=0,
    filenames=False, **kwargs):
    """
    For each of a set of trajectory CSVs, plot the angular
    distribution of displacements.

    args
    ----
        csv_files       :   variadic str, paths to CSV files
        start_frame     :   int, the first frame in the file
                            consider
        out_png         :   str, save path
        kwargs          :   to angular_dist(). These include:
                            min_disp (um), n_bins, norm,
                            pixel_size_um

    returns
    -------
        None

    """
    n = len(csv_files)
    if n == 0:
        print("quot.plot.angular_dist_files: no files passed")
        return 

    m = int(np.ceil(np.sqrt(n)))
    mx = m 
    my = n//m + 1

    fig = plt.figure(figsize=(3*mx, 2*my))
    axes = []
    for i in range(n):
        ax = fig.add_subplot(my, mx, i+1,
            projection="polar")
        axes.append(ax)

    for i, csv_file in enumerate(csv_files):
        tracks = pd.read_csv(csv_file)
        tracks = tracks[tracks["frame"] >= start_frame]
        angular_dist(axes[i], tracks, **kwargs)
        if filenames:
            axes[i].set_title(csv_file, fontsize=10)

    if not out_png is None:
        wrapup(out_png)
    else:
        plt.tight_layout(); plt.show(); plt.close()


#######################
## RELATED UTILITIES ##
#######################

def track_length(tracks):
    """
    Given a set of trajectories in DataFrame format, create
    a new columns ("track_length") with the length of the 
    corresponding trajectory in frames.

    args
    ----
        tracks      :   pandas.DataFrame

    returns
    -------
        pandas.DataFrame, with the new column

    """
    if "track_length" in tracks.columns:
        tracks = tracks.drop("track_length", axis=1)
    tracks = tracks.join(
        tracks.groupby("trajectory").size().rename("track_length"),
        on="trajectory"
    )
    return tracks 

def rad_disp_histogram_2d(tracks, n_frames=4, bin_size=0.001, 
    max_jump=5.0, pixel_size_um=0.160, n_gaps=0, use_entire_track=False,
    max_jumps_per_track=10):
    """
    Compile a histogram of radial displacements in the XY plane for 
    a set of trajectories ("tracks").

    Identical with strobemodels.utils.rad_disp_histogram_2d.

    args
    ----
        tracks          :   pandas.DataFrame
        n_frames        :   int, the number of frame delays to consider.
                            A separate histogram is compiled for each
                            frame delay.
        bin_size        :   float, the size of the bins in um. For typical
                            experiments, this should not be changed because
                            some diffusion models (e.g. Levy flights) are 
                            contingent on the default binning parameters.
        max_jump        :   float, the max radial displacement to consider in 
                            um
        pixel_size_um   :   float, the size of individual pixels in um
        n_gaps          :   int, the number of gaps allowed during tracking
        use_entire_track:   bool, use every displacement in the dataset
        max_jumps_per_track:   int, the maximum number of displacements
                            to consider per trajectory. Ignored if 
                            *use_entire_track* is *True*.

    returns
    -------
        (
            2D ndarray of shape (n_frames, n_bins), the distribution of 
                displacements at each time point;
            1D ndarray of shape (n_bins+1), the edges of each bin in um
        )

    """
    # Sort by trajectory, then frame
    tracks = tracks.sort_values(by=["trajectory", "frame"])

    # Assign track lengths
    if "track_length" not in tracks.columns:
        tracks = track_length(tracks)

    # Filter out unassigned localizations and singlets
    T = tracks[
        np.logical_and(tracks["trajectory"]>=0, tracks["track_length"]>1)
    ][["frame", "trajectory", "y", "x"]]

    # Convert to ndarray for speed
    T = np.asarray(T[["frame", "trajectory", "y", "x", "trajectory"]]).astype(np.float64)

    # Sort first by track, then by frame
    T = T[np.lexsort((T[:,0], T[:,1])), :]

    # Convert from pixels to um
    T[:,2:4] = T[:,2:4] * pixel_size_um 

    # Format output histogram
    bin_edges = np.arange(0.0, max_jump+bin_size, bin_size)
    n_bins = bin_edges.shape[0]-1
    H = np.zeros((n_frames, n_bins), dtype=np.int64)

    # Consider gap frames
    if n_gaps > 0:

        # The maximum trajectory length, including gap frames
        max_len = (n_gaps + 1) * n_frames + 1

        # Consider every shift up to the maximum trajectory length
        for l in range(1, max_len+1):

            # Compute the displacement for all possible jumps
            diff = T[l:,:] - T[:-l,:]

            # Map the trajectory index corresponding to the first point in 
            # each trajectory
            diff[:,4] = T[l:,1]

            # Only consider vectors between points originating from the same track
            diff = diff[diff[:,1] == 0.0, :]

            # Look for jumps corresponding to each frame interval being considered
            for t in range(1, n_frames+1):

                # Find vectors that match the delay being considered
                subdiff = diff[diff[:,0] == t, :]

                # Only consider a finite number of displacements from each trajectory
                if not use_entire_track:
                    _df = pd.DataFrame(subdiff[:,4], columns=["traj"])
                    _df["ones"] = 1
                    _df["index_in_track"] = _df.groupby("traj")["ones"].cumsum() 
                    subdiff = subdiff[np.asarray(_df["index_in_track"]) <= max_jumps_per_track, :]

                # Calculate radial displacements
                r_disps = np.sqrt((subdiff[:,2:4]**2).sum(axis=1))
                H[t-1,:] = H[t-1,:] + np.histogram(r_disps, bins=bin_edges)[0]

    # No gap frames
    else:

        # For each frame interval and each track, calculate the vector change in position
        for t in range(1, n_frames+1):
            diff = T[t:,:] - T[:-t,:]

            # Map trajectory indices back to the first localization of each trajectory
            diff[:,4] = T[t:,1]

            # Only consider vectors between points originating in the same track
            diff = diff[diff[:,1] == 0.0, :]

            # Only consider vectors that match the delay being considered
            diff = diff[diff[:,0] == t, :]

            # Only consider a finite number of displacements from each trajectory
            if not use_entire_track:
                _df = pd.DataFrame(diff[:,4], columns=["traj"])
                _df["ones"] = 1
                _df["index_in_track"] = _df.groupby("traj")["ones"].cumsum()
                diff = diff[np.asarray(_df["index_in_track"]) <= max_jumps_per_track, :]

            # Calculate radial displacements
            r_disps = np.sqrt((diff[:,2:4]**2).sum(axis=1))
            H[t-1,:] = np.histogram(r_disps, bins=bin_edges)[0]

    return H, bin_edges 

def coarsen_histogram(jump_length_histo, bin_edges, factor):
    """
    Given a jump length histogram with many small bins, aggregate into a 
    histogram with a small number of larger bins.

    This is useful for visualization.

    args
    ----
        jump_length_histo       :   2D ndarray, the jump length histograms
                                    indexed by (frame interval, jump length bin)
        bin_edges               :   1D ndarray, the edges of each jump length
                                    bin in *jump_length_histo*
        factor                  :   int, the number of bins in the old histogram
                                    to aggregate for each bin of the new histogram

    returns
    -------
        (
            2D ndarray, the aggregated histogram,
            1D ndarray, the edges of each jump length bin the aggregated histogram
        )

    """
    # Get the new set of bin edges
    n_frames, n_bins_orig = jump_length_histo.shape 
    bin_edges_new = bin_edges[::factor]
    n_bins_new = bin_edges_new.shape[0] - 1

    # May need to truncate the histogram at the very end, if *factor* doesn't
    # go cleanly into the number of bins in the original histogram
    H_old = jump_length_histo[:, (bin_edges<bin_edges_new[-1])[:-1]]

    # Aggregate the histogram
    H = np.zeros((n_frames, n_bins_new), dtype=jump_length_histo.dtype)
    for j in range(factor):
        H = H + H_old[:, j::factor]

    return H, bin_edges_new 

def bond_angles(tracks, min_disp=0.2, delta=1):
    """
    Return the set of angles between jumps originating from 
    trajectories. Angles between pi and 2 * pi are reflected onto
    the interval 0, pi.

    args
    ----
        tracks      :   pandas.DataFrame
        min_disp    :   float, pixels. Discard displacements less than
                        this displacement. This prevents us from
                        being biased by localization error.
        delta       :   int, the proximity of the two jumps with
                        respect to which the angle is calculated.
                        For example, if *delta* is 1, then the angle
                        between subsequent jumps is calculated. 
                        If *delta* is 2, then the angle between 
                        jump n and jump n+2 is calculated, and so on.

    returns
    -------
        1D ndarray of shape (n_angles,), the observed
            angles in radians (from 0 to pi)

    """
    assert delta > 0, "delta must be greater than 0 (passed: {})".format(delta)

    # Get the set of all trajectories with at least three points
    # as an ndarray
    tracks = track_length(tracks)
    T = np.asarray(
        tracks[(tracks["trajectory"] >= 0) & (tracks["track_length"] > (1+delta))][
            ["trajectory", "frame", "y", "x"]
        ]
    )
    if T.shape[0] == 0:
        return np.nan

    # Ascending trajectory and frame indices
    tracks = tracks.sort_values(by=["trajectory", "frame"])

    # Unique trajectory indices
    traj_indices = np.unique(T[:, 0])

    # Largest number of possible angles
    n_angles = T.shape[0] - 2 * len(traj_indices)

    # Output array
    angles = np.zeros(n_angles, dtype="float64")

    # Iterate through all trajectories
    c = 0
    for i, j in enumerate(traj_indices):

        # The set of y, x points corresponding to this trajectory
        traj = T[T[:, 0] == j, 2:]

        # Jumps between subsequent frames
        disps = traj[1:, :] - traj[:-1, :]

        # Jump moduli
        mags = np.sqrt((disps ** 2).sum(axis=1))

        # Angles between subsequent jumps
        traj_angles = (disps[delta:, :] * disps[:-delta, :]).sum(axis=1) / (mags[delta:] * mags[:-delta])

        # Only take angles above a given displacement, if desired
        traj_angles = traj_angles[(mags[delta:] >= min_disp) & (mags[:-delta] >= min_disp)]

        # Aggregate
        n_traj_angles = traj_angles.shape[0]
        angles[c : c + n_traj_angles] = traj_angles
        c += n_traj_angles

    # We'll lose some angles because of the min_disp cutoff
    angles = angles[:c]

    # Some floating point errors occur here - values slightly
    # greater than 1.0 or less than -1.0
    angles[angles > 1.0] = 1.0 
    angles[angles < -1.0] = -1.0 

    return np.arccos(angles[~pd.isnull(angles)])
