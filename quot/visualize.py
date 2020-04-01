"""
visualize.py -- plots for quot

"""
# Numerics / dataframes
import numpy as np 
import tifffile 
import pandas as pd 
from quot import qio 

# File paths
import os
import sys 

# Plotting
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(style='ticks')

# Progress bar 
from tqdm import tqdm 

def wrapup(out_png, dpi=500):
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()
    os.system('open %s' % out_png)

def loc_density(
    locs,
    image_height,
    image_width,
    ax=None,
    pixel_size_um=0.16,
    upsampling_factor=20,
    kernel_width=0.5,
    y_col='y0',
    x_col='x0',
    vmax_mod=1.0,
    cmap='gray',
    plot=False,
): 
    # Get the list of localized positions
    positions = np.asarray(locs[[y_col, x_col]]) * pixel_size_um 

    # Make the size of the out frame
    N = image_height
    M = image_width 
    n_up = int(N * pixel_size_um * upsampling_factor)
    m_up = int(M * pixel_size_um * upsampling_factor)

    # The density
    density = np.zeros((n_up, m_up), dtype = 'float64')

    # Determine the size of the Gaussian kernel to use for
    # KDE
    sigma = kernel_width * upsampling_factor
    w = int(6 * sigma)
    if w % 2 == 0: w+=1
    half_w = w // 2
    r2 = sigma ** 2
    kernel_y, kernel_x = np.mgrid[:w, :w]
    kernel = np.exp(-((kernel_x-half_w)**2 + (kernel_y-half_w)**2) / (2*r2))

    # Make the kernel density estimate 
    n_locs = len(locs)
    for loc_idx in range(n_locs):
        y = int(round(positions[loc_idx, 0] * upsampling_factor, 0))
        x = int(round(positions[loc_idx, 1] * upsampling_factor, 0))
        try:
            # Localization is entirely inside the borders
            density[
                y-half_w : y+half_w+1,
                x-half_w : x+half_w+1,
            ] += kernel
        except ValueError:
            # Localization is close to the edge
            k_y, k_x = np.mgrid[y-half_w:y+half_w+1, x-half_w:x+half_w+1]
            in_y, in_x = ((k_y>=0) & (k_x>=0) & (k_y<n_up) & (k_x<m_up)).nonzero()
            density[k_y.flatten()[in_y], k_x.flatten()[in_x]] = \
                density[k_y.flatten()[in_y], k_x.flatten()[in_x]] + kernel[in_y, in_x]

    # Plot the result
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(density[::-1,:], cmap=cmap, 
                vmax=density.mean()+density.std()*vmax_mod)       
            plt.show(); plt.close()
        else:
            ax.imshow(density[::-1,:], cmap=cmap, 
                vmax=density.mean()+density.std()*vmax_mod)
    return density 

def attrib_dist_2d(locs, attrib_0, attrib_1, out='default.png',
    **kwargs):
    """
    Make a 2D histogram of the values of two different
    parameters across all of the localizations.

    args
    ----
        locs : pandas.DataFrame
        attrib_0 : str, a column in locs
        attrib_1 : str, a column in locs 

    """
    assert attrib_0 in locs.columns
    assert attrib_1 in locs.columns 
    X = np.asarray(locs[[attrib_0, attrib_1]])

    # Choose the limits on the graph
    y0 = max([
        X[:,0].mean() - 3*X[:,0].std(),
        X[:,0].min()
    ])
    y1 = min([
        X[:,0].mean() + 3*X[:,0].std(),
        X[:,0].max()
    ])
    x0 = max([
        X[:,1].mean() - 3*X[:,1].std(),
        X[:,1].min()
    ])
    x1 = min([
        X[:,1].mean() + 3*X[:,1].std(),
        X[:,1].max()
    ])

    unit_sizes = np.array([y1-y0, x1-x0])*3/100

    # Get the local density of datapoints
    density = n_neighbors(X)

    # Truncate the density at the 90% percentile,
    # to avoid outliers dominating the colormap
    p = np.percentile(density, 90)
    density[density>p] = p 

    # Plot 
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(locs[attrib_0], locs[attrib_1], 
        c=density, s=5, cmap='magma', **kwargs)

    # Set axis labels
    ax.set_xlabel(attrib_0, fontsize=14)
    ax.set_ylabel(attrib_1, fontsize=14)

    # Set limits
    ax.set_ylim((x0, x1))
    ax.set_xlim((y0, y1))

    # Save
    plt.tight_layout()
    plt.savefig(out, dpi=600)
    os.system('open %s' % out)

def n_neighbors(X, unit_sizes=None):
    """
    Return the relative density of observations of 
    an M-dimensional vector.

    args
    ----
        X : 2D ndarray of shape (n_points, M),
            M : dimensions

    returns
    -------
        1D ndarray of shape (n_points,), the number
            of neighbors of each observation

    """
    n = X.shape[0]

    # Unit size in each dimension
    if unit_sizes is None:
        unit_sizes = X.std(axis=0)*8/100

    # Normalize the array
    X_norm = X / unit_sizes 

    # For each observation, compute the number of 
    # observations close to that observation
    result = np.zeros(n, dtype='float64')
    for i in range(n):
        result[i] = (((X_norm-X_norm[i,:])**2).sum(axis=1)<=1.0).sum()

    return result

def localization_summary(
    locs,
    image_file_path,
    psf_window_size=11,
    psf_distance_from_center=0.25,
    loc_density_upsampling_factor=10,
    y_col='y0',
    x_col='x0',
    pixel_size_um=0.16,
    out_png=None,
):
    '''
    Make a panel of plots related to the quality of the 
    localizations. 

    args
        locs            :       pandas.DataFrame
        psf_window_size                 :       int
        psf_distance_from_center        :   float
        loc_density_upsampling_factor   :   int
        y_col           :       str, column for y position in *locs*
        x_col           :       str, column for x position in *locs*
        out_png         :       str, file to save plot to

    '''
    reader = qio.ImageFileReader(image_file_path)
    n_frames, N, M = reader.get_shape()

    if 'err_y0' in locs.columns:
        fig, ax = plt.subplots(3, 3, figsize = (9, 9))
    else:
        fig, ax = plt.subplots(2, 3, figsize = (9, 6))

    # Show the mean PSF 
    try:
        print('Compiling mean PSF...')
        psf = calculate_psf(
            image_file_path,
            locs,
            window_size = psf_window_size,
            distance_from_center = psf_distance_from_center,
            y_col = y_col,
            x_col = x_col,
        )
        ax[0,0].imshow(psf, cmap='inferno')
        ax[0,0].set_xticks([])
        ax[0,0].set_yticks([])
        ax[0,0].set_title('Mean PSF')
        ax[0,0].set_aspect('equal')
    except FileNotFoundError:
        print('Could not find file %s for the PSF calculation' % nd2_file)
        ax[0,0].grid(False)
        for spine_dir in ['top', 'bottom', 'left', 'right']:
            ax[0,0].spines[spine_dir].set_visible(False)
        ax[0,0].set_xticks([])
        ax[0,0].set_yticks([])

    # Show the distribution of photon counts per localization
    attrib_dist(
        locs,
        ax = ax[0,1],
        attrib = 'I',
        label = 'Photon count',
        color = '#C2C2C2',
        max_value = 1000,
        bin_size = 40,
    )

    # Show the distribution of photon counts as a function of
    # space
    show_locs(
        locs,
        ax = ax[0,2],
        color_attrib = 'I',
        cmap = 'viridis',
        max_value = 500,
        min_value = 0,
        ylim = ((0, N)),
        xlim = ((0, M)),
    )
    ax[0,2].set_title('Photon counts')
    ax[0,2].set_aspect('equal')

    # Show the localization density
    density = loc_density(locs, N, M, ax=ax[1,0],
        upsampling_factor=loc_density_upsampling_factor,
        kernel_width=0.1, y_col=y_col, x_col=x_col, 
        vmax_mod=1.5, cmap='gray', plot=True, 
        pixel_size_um=pixel_size_um)
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,0].set_title("Localization density")
    ax[1,0].set_aspect('equal')

    # Show the pixel localization density
    pixel_density, y_field, x_field = pixel_localization_density(
        locs,
        bin_size = 0.05,
        y_col = y_col,
        x_col = x_col,
        plot = False,
    )
    ax[1,1].imshow(pixel_density[::-1,:], cmap = 'gray', vmin = 0, vmax = pixel_density.max())
    ax[1,1].set_xticks([-0.5, 4.5, 9.5, 14.5, 19.5])
    ax[1,1].set_xticklabels([0.00, 0.25, 0.50, 0.75, 1.00])
    ax[1,1].set_yticks([-0.5, 4.5, 9.5, 14.5, 19.5])
    ax[1,1].set_yticklabels([0.00, 0.25, 0.50, 0.75, 1.00])
    ax[1,1].set_xlabel('x mod 1 (pixels)')
    ax[1,1].set_ylabel('y mod 1 (pixels)')
    ax[1,1].set_title('Pixel loc density')
    ax[1,1].set_aspect('equal')

    # Show the number of localizations per frame
    locs_per_frame, interval_times = loc_density_time(locs, ax=ax[1,2])

    if 'err_y0' in locs.columns:

        print(locs.columns)

        # Show localization error as a function of space
        new_locs = locs.assign(average_error=locs[['err_y0','err_x0']].mean(axis=1)*160)
        new_locs = new_locs[new_locs['average_error'] <= 500]
        new_locs = new_locs[new_locs['average_error'] > 0.0]
        show_locs(
            new_locs,
            ax = ax[2,0],
            color_attrib = 'average_error',
            cmap = 'inferno',
            max_value = 150,
            min_value = 0,
            ylim = ((0, N)),
            xlim = ((0, M)),
        )
        ax[2,0].set_title('Localization error')

        # Show distribution of pixel localization errors
        attrib_dist(
            new_locs,
            ax = ax[2,1],
            attrib = 'average_error',
            color = '#C2C2C2',
            max_value = 200,
            bin_size = 10,
        )
        ax[2,1].set_xlabel('Est. localization error (nm)')
        ax[2,1].set_ylabel('Localizations')

        # Show distribution of I errors
        attrib_dist(
            locs,
            ax = ax[2,2],
            attrib = 'I',
            color = '#C2C2C2',
            max_value = 100,
            bin_size = 5,
        )
        ax[2,2].set_xlabel('Est. I error (photons)')
        ax[2,2].set_ylabel('Localizations')

    # Save to a PNG, if desired
    plt.tight_layout()
    plt.savefig(out_png, dpi = 600)
    if sys.platform == 'darwin':
        os.system('open %s' % out_png)

#
# Functions for localization_qc
#
def attrib_dist(locs, ax=None, attrib='I', color='#C2C2C2',
    bin_size=20, max_value=500, label=None):
    """
    One-dimensional histogram of an attribute.

    """
    bin_edges = np.arange(0, max_value+bin_size, bin_size)
    histo, edges = np.histogram(locs[attrib], bins=bin_edges)
    bin_centers = bin_edges[:-1] + bin_size/2

    if ax == None:
        fig, ax = plt.subplots(figsize = (3, 2))
        finish_plot = True 
    else:
        finish_plot = False 

    ax.bar(
        bin_centers,
        histo,
        width=bin_size * 0.8,
        edgecolor='k',
        linewidth=2,
        color=color,
    )
    if label == None:
        ax.set_xlabel(attrib)
    else:
        ax.set_xlabel(label)
    ax.set_ylabel('Localizations')

    if finish_plot:
        plt.tight_layout(); plt.show(); plt.close()

def loc_density_time(locs, ax=None, frame_interval=5,
    frame_col='frame_idx'):
    """
    Plot localization density as a function of 
    frame index.

    """
    new_locs = locs.assign(frame_group=locs[frame_col]//frame_interval)
    locs_per_frame = new_locs.groupby('frame_group').size()/frame_interval 

    if ax == None:
        fig, ax = plt.subplots(figsize = (3, 3))
        finish_plot = True
    else:
        finish_plot = False 

    interval_times = np.arange(len(locs_per_frame))*frame_interval
    ax.plot(
        interval_times,
        locs_per_frame,
        color = 'k',
        linestyle = '-',
        linewidth = 2,
    )
    ax.set_xlabel('Frame index')
    ax.set_ylabel('Localizations per frame')

    if finish_plot:
        plt.tight_layout()
        plt.show(); plt.close()

    return locs_per_frame, interval_times

def pixel_localization_density(locs, bin_size=0.05, 
    y_col='y0', x_col='x0', plot=False):
    '''
    Calculate the pixel localization density as a function 
    of position along each camera pixel, to check for pixel
    edge bias.

    args
        locs        :   pandas.DataFrame
        bin_size    :   float, the fraction of a pixel at   
                        which to make bins
        y_col       :   str, colunn in *locs* with y-coordinate
        x_col       :   str, column in *locs* with x-coordinate

    returns
        2D ndarray of shape (n_bins, n_bins), the 2D histogram
            of localization counts in each spatial bin

    '''
    bin_seq = np.arange(0, 1.0+bin_size, bin_size)
    n_bins = len(bin_seq) - 1

    histo, xedges, yedges = np.histogram2d(
        locs[y_col] % 1.0,
        locs[x_col] % 1.0,
        bins = bin_seq,
    )
    y_field, x_field = np.mgrid[:n_bins, :n_bins] / n_bins + bin_size/2

    if plot:
        y_field, x_field = np.mgrid[:n_bins, :n_bins] / n_bins + bin_size/2
        fig = plt.figure(figsize = (6, 4))
        ax = fig.add_subplot(111, projection = '3d')
        ax.plot_surface(
            y_field,
            x_field,
            histo,
            cmap = 'inferno',
            linewidth = 0.1,
        )
        ax.set_zlabel('Localization count')
        ax.set_xlabel('x mod 1 (pixels)')
        ax.set_ylabel('y mod 1 (pixels)')
        plt.show(); plt.close()

    return histo, y_field, x_field 

def calculate_psf(
    image_file,
    locs,
    window_size = 11,
    frame_col = 'frame_idx',
    y_col = 'y0',
    x_col = 'x0',
    distance_from_center = 0.5,
    plot = False,
):
    '''
    Calculate the mean PSF for a set of localizations.

    args
        image_file              :   str, either TIF or ND2

        locs                    :   pandas.DataFrame, localizations

        window_size             :   int, the size of the PSF image window

        frame_col, y_col, x_col :   str, column names in *locs*

        distance_from_center    :   float, the maximum tolerated distance
                                        from the center of the pixel along
                                        either the y or x directions. If 0.5,
                                        then all localizations are used (even
                                        if they fall on a corner). If 0.25, then
                                        only pixels in the range [0.25, 0.75]
                                        on each pixel would be used, etc.

    returns
        2D ndarray of shape (window_size, window_size), the averaged
            PSF image 

    '''
    in_radius = (((locs[y_col]%1.0))**2 + ((locs[x_col]%1.0))**2) <= (distance_from_center**2)
    select_locs = locs[in_radius]

    reader = qio.ImageFileReader(image_file)
    psf_image = np.zeros((window_size, window_size), dtype = 'float64')
    n_psfs = 0
    half_w = window_size // 2

    for frame_idx, frame_locs in tqdm(select_locs.groupby(frame_col)):
        image = reader.get_frame(frame_idx)
        for loc_idx in frame_locs.index:
            y, x = np.asarray(frame_locs.loc[loc_idx, [y_col, x_col]]).astype('uint16')
            try:
                subim = image[
                    y-half_w : y+half_w+1,
                    x-half_w : x+half_w+1,
                ]
                psf_image = psf_image + subim
                n_psfs += 1
            except ValueError: #edge loc 
                pass 

    psf_image = psf_image / n_psfs 
    reader.close()
    if plot: plt.imshow(psf_image); plt.show(); plt.close()
    return psf_image 

def show_locs(
    locs,
    ax = None,
    color_attrib = 'I',
    cmap = 'inferno',
    max_value = 500,
    min_value = 0,
    ylim = None,
    xlim = None,
    y_col='y0',
    x_col='x0',
    aspect_equal=True,
):
    if ax == None:
        fig, ax = plt.subplots(figsize = (6, 6))
        finish_plot = True 
    else:
        finish_plot = False 

    color_index = locs[color_attrib].copy()
    color_index = (color_index - min_value)
    color_index[color_index > max_value] = max_value

    ax.scatter(
        locs[x_col],
        locs[y_col],
        c = color_index,
        cmap = cmap,
        s = 2,
    )
    if type(ylim) == type((0,0)):
        ax.set_ylim(ylim)
    if type(xlim) == type((0,0)):
        ax.set_xlim(xlim)

    if aspect_equal:
        ax.set_aspect('equal')

    if finish_plot:
        plt.show(); plt.close()

# 
# Masking QC
#
def mask_summary(path, locs, height, width, out_png=None,
    verbose=True, y_col='y0', x_col='x0'):
    """
    Display a variety of plots related to masking.

    ax[0,0] -> simply show the mask 
    ax[0,1] -> show the localization density
    ax[1,0] -> show the locs inside the mask only
    ax[1,1] -> show all locs, colored by mask membership

    """
    fig, ax = plt.subplots(2, 2, figsize=(6, 6))

    # Calculate localization density, for comparison
    locs['loc_density'] = n_neighbors(np.asarray(locs[['y0', 'x0']]))

    # Get the localizations inside and outside the
    # mask
    in_mask = path.contains_points(np.asarray(locs[['x0', 'y0']]))
    locs_in_mask = locs[in_mask]
    locs_out_mask = locs[~in_mask]

    # ax[0,0]: show the mask 
    Y, X = np.indices((height*2, width*2)) * 0.5
    yx = np.asarray([Y.flatten(), X.flatten()]).T 
    binary_mask = path.contains_points(yx).reshape((height*2, width*2))
    ax[0,0].imshow(binary_mask[:,::-1].T, cmap='gray')

    # ax[0,1]: show the localization density
    img_loc_density = loc_density(locs, height, width, ax=ax[0,1],
        pixel_size_um=1.0, upsampling_factor=10, kernel_width=1.0,
        vmax_mod=1.5, cmap='gray', plot=True)

    # ax[1,0]: show all of the masked localizations
    show_locs(locs_in_mask, ax=ax[1,0], color_attrib='loc_density',
        cmap='viridis', max_value=500)

    # ax[1,1]: show both masked and unmasked localizations
    show_locs(locs_in_mask, ax=ax[1,1], color_attrib='loc_density',
        cmap='viridis', max_value=500)   
    show_locs(locs_out_mask, ax=ax[1,1], color_attrib='loc_density',
        cmap='inferno', max_value=500)   

    # Set plot limits
    ax[1,0].set_xlim((0, width))
    ax[1,0].set_ylim((0, height))
    ax[1,1].set_xlim((0, width))
    ax[1,1].set_ylim((0, height))

    # Set subplot titles
    ax[0,0].set_title('Mask definition')
    ax[0,1].set_title('Localization density')
    ax[1,0].set_title('Locs in mask')
    ax[1,1].set_title('Mask membership')

    # Kill the ticks
    for i in range(2):
        for j in range(2):
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

    plt.tight_layout()
    if type(out_png) == type(''):
        plt.savefig(out_png, dpi=600)
        if sys.platform == 'darwin':
            os.system('open %s' % out_png)
    else:
        plt.show()

def attribute_mask_summary(path, locs, edges_0, edges_1,
    attrib_0, attrib_1, xlabel=None, ylabel=None,
    out_png='default_attribute_mask_summary.png'):
    """
    Show a variety of plots related to attribute masking.

    ax[0] -> the raw binary mask
    ax[1] -> only the masked part of the histogram
    ax[2] -> both masked and unmasked parts of the histogram

    args
    ----
        path : matplotlib.path.Path object, the path of the
                mask
        edges_0 : 1D ndarray, edges along y axis
        edges_1 : 1D ndarray, edges along x axis
        out_png : str 

    """
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))

    # Make histogram
    data = np.asarray(locs[[attrib_0, attrib_1]])
    H, _edges_0, _edges_1 = np.histogram2d(data[:,0],
        data[:,1], bins=(edges_0, edges_1))

    # Evaluate density
    density = n_neighbors(data)

    # Density calculation
    locs['attrib_density'] = n_neighbors(data)

    # ax[0]: raw mask 
    N = edges_0.shape[0]-1
    M = edges_1.shape[0]-1
    Y, X = np.indices((N,M))
    Y = (Y*(edges_0[1]-edges_0[0])) + edges_0.min()
    X = (X*(edges_1[1]-edges_1[0])) + edges_1.min()
    YX = np.asarray([Y.ravel(), X.ravel()]).T 
    mask_def = path.contains_points(YX).reshape((N, M))
    ax[0].imshow(mask_def[:,::-1].T, cmap='gray')

    # ax[1] : the actual histogram
    ax[1].imshow(H[:,::-1].T, cmap='viridis')

    # ax[2] : all data points
    in_mask = path.contains_points(data)
    locs_in_mask = locs[in_mask]
    locs_out_mask = locs[~in_mask]
    ax[2].scatter(locs_out_mask[attrib_0], locs_out_mask[attrib_1],
        cmap='inferno', c=density[~in_mask], s=1)
    ax[2].scatter(locs_in_mask[attrib_0], locs_in_mask[attrib_1],
        cmap='viridis', c=density[in_mask], s=1)

    # Plot limits
    ax[2].set_xlim((edges_0.min(), edges_0.max()))
    ax[2].set_ylim((edges_1.min(), edges_1.max()))

    # Plot titles
    ax[0].set_title("Mask definition")
    ax[1].set_title("Histogram")
    ax[2].set_title("Datapoints")

    # Labels
    if not xlabel is None:
        for j in range(3): ax[j].set_xlabel(xlabel)
    if not ylabel is None:
        for j in range(3): ax[j].set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    if sys.platform == 'darwin':
        os.system('open %s' % out_png)

