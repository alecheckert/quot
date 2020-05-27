#!/usr/bin/env python
"""
plot.py -- simple plotting utilities for quot

"""
# Numeric
import numpy as np 

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

###############################
## TRAJECTORY ANALYSIS PLOTS ##
###############################

def plotRadialDisps(radial_disps, bin_edges, frame_interval=0.00548, plot=True):
    """
    Plot a set of radial displacement histograms.

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
    bin_c = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2.0

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
        ax[interval].plot(bin_c, pmf, color=colors[interval],
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




