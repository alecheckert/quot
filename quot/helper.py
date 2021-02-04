#!/usr/bin/env python
"""
quot.helper.py -- low-level utilities

"""
# Paths
import os
from glob import glob 

# Numeric
import numpy as np 

# Dataframes
import pandas as pd 

# Special functions
from scipy.special import erf 

# Image processing utilities
from scipy import ndimage as ndi 

# Load *.mat format files
from scipy import io as sio 

# Caching
from functools import lru_cache 

# Warnings, to filter out warnings for a potentially dangerous
# division in rs() that is subsequent corrected
import warnings 

# Profiling
from time import time 

from .plot import wireframe_overlay

def time_f(f):
    """
    Decorator. When executing the decorated function,
    also print the time to execute.

    args
    ----
        f   :   function

    returns
    -------
        output of f

    """
    def g(*args, **kwargs):
        t0 = time()
        r = f(*args, **kwargs)
        t1 = time()
        print("%f seconds" % (t1-t0))
        return r 
    return g

#############
## GENERAL ##
#############

def assign_methods(methods):
    """
    Decorator that designates a wrapper function for 
    other methods. Adds a new kwarg to the decorated 
    function ("method") that looks up the corresponding
    method in the *methods* dict.

    Example:
        def add(x, y):
            return x + y
        def mult(x, y):
            return x * y
        METHODS = {'add': add, 'mult': mult}

        do_operation = assign_methods(METHODS)(lambda a, b: None)

    This would be called:
        do_operation(2, 3, method='add')  # is 5
        do_operation(2, 3, method='mult') # is 6

    args
    ----
        methods     :   dict that keys strings to methods

    returns
    -------
        function, wrapper for the methods

    """ 
    def dec(f):
        def apply_method(*args, method=None, **kwargs):
            print(method)
            method_f = methods.get(method)
            if method_f is None:
                raise KeyError("Method %s not found; available methods are %s" % (
                    method, ", ".join(methods.keys())))
            return method_f(*args, **kwargs)
        return apply_method
    return dec

def stable_divide_array(N, D, zero=0.001):
    """
    Divide array N by array D, setting zeros in D
    to *zero*. Assumes nonzero D.

    args
    ----
        N, D    :   ndarrays of the same shape
        zero    :   float

    returns
    -------
        ndarray of the same shape as N and D

    """
    return N/np.clip(D, zero, np.inf)

def stable_divide_float(N, D, inf=0.0):
    """
    Divide float N by float D, returning *inf*
    if *D* is zero.

    args
    ----
        N, D    :   float
        inf     :   float

    returns
    -------
        float

    """
    return (inf if D==0.0 else N/D)

#####################
## FILE CONVERSION ##
#####################

def tracked_mat_to_csv(path, out_csv=None, pixel_size_um=0.16):
    """
    Convert a file from *Tracked.mat format, a MATLAB-based
    format for trajectories to a DataFrame format.

    args
    ----
        path        :   str, path to a *Tracked.mat file
        out_csv     :   str, file to save the result to 
        pixel_size_um:  float, the size of pixels in um

    returns
    -------
        pandas.DataFrame, the contents of the *Tracked.mat 
            file as a DataFrame

    """
    f = sio.loadmat(path)['trackedPar']
    if len(f.shape) == 2 and f.shape[0] == 1:
        f = f[0,:]

    # Total number of localizations in the file
    n_locs = sum([f[i][0].shape[0] for i in range(len(f))])

    # Output dataframe
    df = pd.DataFrame(index=np.arange(n_locs),
        columns=["y_um", "x_um", "frame", "trajectory", "time"])

    # Trajectory indices
    df["trajectory"] = np.concatenate([[i for j in range(f[i][0].shape[0])] for i in range(len(f))])

    # XY positions of each localization
    df[["y_um", "x_um"]] = np.concatenate([f[i][0] for i in range(len(f))], axis=0)

    # Frame index (correct for MATLAB off-by-1)
    df["frame"] = np.concatenate([f[i][1][:,0] for i in range(len(f))]) - 1

    # Timepoint in seconds
    df["time"] = np.concatenate([f[i][2][:,0] for i in range(len(f))])

    # Convert XY positions to pixels and correct for the MATLAB
    # off-by-1
    df[["y", "x"]] = (df[["y_um", "x_um"]] / pixel_size_um) - 1

    # Assign other columns that may be expected by other utilities
    df["error_flag"] = 0.0

    if not out_csv is None:
        df.to_csv(out_csv, index=False)

    return df

def load_tracks_dir(dirname, suffix="trajs.csv", start_frame=0,
    min_track_len=1):
    """
    Load all of the trajectory files in a target directory
    into a single pandas.DataFrame.

    args
    ----
        dirname         :   str, directory containing track CSVs
        suffix          :   str, extension for the track CSVs
        start_frame     :   int, exclude all trajectories before
                            this frame
        min_track_len   :   int, the minimum trajectory length to 
                            include

    returns
    -------
        pandas.DataFrame

    """
    # Find target files
    if os.path.isdir(dirname):
        target_csvs = glob(os.path.join(dirname, "*{}".format(suffix)))
        if len(target_csvs) == 0:
            raise RuntimeError("quot.helper.load_tracks_dir: could not find " \
                "trajectory CSVs in directory {}".format(dirname))
    elif os.path.isfile(dirname):
        target_csvs = [dirname]

    # Concatenate trajectories
    tracks = [pd.read_csv(j) for j in target_csvs]
    tracks = concat_tracks(*tracks)

    # Exclude points before the start frame
    tracks = tracks[tracks["frame"] >= start_frame]

    # Exclude trajectories that are too short
    tracks = track_length(tracks)
    if min_track_len > 1:
        tracks = tracks[tracks["track_length"] >= min_track_len]

    return tracks 


###############
## DETECTION ##
###############

def pad(I, H, W, mode='ceil'):
    """
    Pad an array with zeroes around the edges, placing 
    the original array in the center. 

    args
    ----
        I       :   2D ndarray, image to be padded
        H       :   int, height of final image
        W       :   int, width of final image
        mode    :   str, either 'ceil' or 'floor'. 'ceil'
                    yields convolution kernels that function
                    similarly to scipy.ndimage.uniform_filter.

    returns
    -------
        2D ndarray, shape (H, W)

    """
    H_in, W_in = I.shape
    out = np.zeros((H, W))
    if mode == "ceil":
        hc = np.ceil(H / 2 - H_in / 2).astype(int)
        wc = np.ceil(W / 2 - W_in / 2).astype(int)
    elif mode == "floor":
        hc = np.floor(H / 2 - H_in / 2).astype(int)
        wc = np.floor(W / 2 - W_in / 2).astype(int)
    out[hc : hc + H_in, wc : wc + W_in] = I
    return out

def label_spots(binary_img, intensity_img=None, mode="max"):
    """
    Find continuous nonzero objects in a binary image,
    returning the coordinates of the spots.

    If the objects are larger than a single pixel,
    then to find the central pixel do
        1. use the center of mass (if mode == 'centroid')
        2. use the brightest pixel (if mode == 'max')
        3. use the mean position of the binary spot
            (if img_int is not specified)

    args
    ----
        binary_img      :   2D ndarray (YX), dtype bool
        intensity_img   :   2D ndarray (YX)
        mode            :   str, 'max' or 'centroid'

    returns
    -------
        2D ndarray (n_spots, 2), dtype int64,
            the Y and X coordinate of each spot

    """
    # Find and label every nonzero object
    img_lab, n = ndi.label(binary_img, structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
    index = np.arange(1, n + 1)

    # Find the centers of the spots
    if intensity_img is None:
        positions = np.asarray(ndi.center_of_mass(binary_img, 
            labels=img_lab, index=index))
    elif mode == "max":
        positions = np.asarray(ndi.maximum_position(intensity_img,
            labels=img_lab, index=index))
    elif mode == "centroid":
        positions = np.asarray(ndi.center_of_mass(intensity_img,
            labels=img_lab, index=index))

    return positions.astype("int64")

def threshold_image(I, t=200.0, return_filt=False, mode='max'):
    """
    Take all spots in an image above a threshold *t*
    and return their approximate centers.

    If *return_filt* is set, the function also returns the 
    raw image and binary image. This is useful as a 
    back door when writing GUIs.

    args
    ----
        I           :   2D ndarray, image
        t           :   float, threshold
        return_filt :   bool
        mode        :   str, either 'max' or 'centroid'

    returns
    -------
        If *return_filt* is set:
        (
            2D ndarray, same as I;
            2D ndarray, binary image;
            2D ndarray, shape (n_spots, 2), the spot
                coordinates
        )
        else
            2D ndarray of shape (n_spots, 2), 
                the spot coordinates

    """
    I_bin = I > t
    pos = label_spots(I_bin, intensity_img=I, mode=mode)
    if return_filt:
        return I, I_bin, pos 
    else:
        return pos

def hollow_box_var(I, w0=15, w1=9):
    """
    Calculate the local variance of every point in a 2D image,
    returning another 2D image composed of the variances.

    Variances are calculated in a "hollow box" region around
    each point. The box is a square kernel of width *w0*; the
    hollow central region is width *w1*.

    For instance, if w0 = 5 and w1 = 3, the kernel would be

            1   1   1   1   1
            1   0   0   0   1
            1   0   0   0   1
            1   0   0   0   1
            1   1   1   1   1

    args
    ----
        I       :   2D ndarray, image
        w0      :   int, width of the kernel
        w1      :   int, width of central region

    returns
    -------
        2D ndarray

    """
    # Make the hollow box kernel
    hw0 = w0 // 2
    hw1 = w1 // 2
    kernel = np.ones((w0, w0), dtype='float64')
    kernel[hw0-hw1:hw0+hw1+1, hw0-hw1:hw0+hw1+1] = 0
    kernel = kernel / kernel.sum()

    # Calculate variance
    kernel_rft = np.fft.rfft2(pad(kernel, *I.shape))
    I_mean = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(I) * kernel_rft, s=I.shape))
    I2_mean = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(I**2) * kernel_rft, s=I.shape))
    return I2_mean - (I_mean**2)



###########################
## SUBPIXEL LOCALIZATION ##
###########################

## LOW-LEVEL SUBPIXEL LOCALIZATION FUNCTIONS

def ring_mean(I):
    """
    Take the mean of the outer ring of pixels in 
    a 2D array.

    args
    ----
        I         :   2D ndarray (YX)

    returns
    -------
        float, mean

    """
    return sum([
        I[0,:-1].mean(), 
        I[:-1,-1].mean(),
        I[-1,1:].mean(),
        I[1:,0].mean(),
    ])/4.0

def ring_var(I, ddof=0):
    """
    Take the variance of the outer ring of pixels in
    a 2D ndarray.

    args
    ----
        I       :   2D ndarray, image
        ddof    :   int, delta degrees of freedom

    returns
    -------
        float, variance estimate

    """
    return np.concatenate([
        I[0,1:],
        I[1:,-1],
        I[-1,:-1],
        I[:-1,0]
    ]).var(ddof=ddof)

def I0_is_crazy(I0, max_I0=10000):
    """
    Determine whether the intensity estimate in subpixel
    localization looks crazy. For many methods, this is 
    often the first thing to diverge.

    args
    ----
        I0  :       float, the intensity estimate for 
                    subpixel fitting
        max_I0  :   float, the maximum tolerated intensity

    returns
    -------
        bool, True if crazy 

    """
    return (I0>=0) and (I0<=max_I0)

def estimate_I0(I, y, x, bg, sigma=1.0):
    """
    Estimate the integrated intensity of a 2D integrated 
    Gaussian PSF. This model has four parameters - y, x,
    I0, and bg - so with y, x, and bg constrained, we can 
    solve for I0 directly.

    In this method, we use the brightest pixel in 
    the image to solve for I0 given the PSF model.
    This is easy but often an overestimate in the presence
    of Poisson noise.

    args
    ----
        I       :   2D ndarray, the PSF image
        y, x    :   float, center of PSF in y and x
        bg      :   float, estimated background intensity
                        per pixel (AU)
        sigma   :   float, 2D integrated Gaussian PSF 
                        radius

    returns
    -------
        float, estimate for I0

    """
    # Get the index of the brightest pixel
    ym, xm = np.unravel_index(np.argmax(I), I.shape)

    # Evaluate the intensity estimate
    return stable_divide_float(
        I[ym,xm] - bg,
        int_gauss_psf_1d(ym, y, sigma=sigma) * \
            int_gauss_psf_1d(xm, x, sigma=sigma),
        inf=np.nan 
    )

def estimate_I0_multiple_points(I, y, x, bg, sigma=1.0):
    """
    Estimate the integrated intensity of a 2D integrated 
    Gaussian PSF. This model has four parameters - y, x,
    I0, and bg - so with y, x, and bg constrained, we can 
    solve for I0 directly.

    In this method, we use the intensities at the 3x3 
    central points in the image window to obtain nine 
    estimates for I0, then average them for the final 
    estimate. This is more accurate than estimate_I0()
    but is slower. Probably overkill for a first guess.

    args
    ----
        I       :   2D ndarray, the PSF image
        y, x    :   float, center of PSF in y and x
        bg      :   float, estimated background intensity
                        per pixel (AU)
        sigma   :   float, 2D integrated Gaussian PSF 
                        radius

    returns
    -------
        float, estimate for I0

    """
    # Get the three central points in the spot window
    # and their coordinates
    wy, wx = I.shape
    hwy = wy//2
    hwx = wx//2
    Y, X = indices(wy, wx)
    sy = slice(hwy-1, hwy+2)
    sx = slice(hwx-1, hwx+2)
    obs_I = I[sy, sx].ravel()
    cen_y = Y[sy, sx].ravel()
    cen_x = X[sy, sx].ravel()

    return stable_divide_array(
        obs_I - bg,
        int_gauss_psf_1d(cen_y, y, sigma=sigma) * \
            int_gauss_psf_1d(cen_x, x, sigma=sigma)
    ).mean()

def amp_from_I(I0, sigma=1.0):
    """
    Given a 2D Gaussian PSF, return the PSF
    peak amplitude given the intensity `I0`.
    `I0` is equal to the PSF integrated above
    background, while `amp` is equal to the PSF
    evaluated at its maximum.

    args
    ----
         I0      : float, intensity estimate
         sigma   : float, width of Gaussian

    returns
    -------
        float, amplitude estimate

    """
    return I0/(2*np.pi*(sigma**2))

def estimate_snr(I, I0):
    """
    Estimate the signal-to-noise ratio of a PSF, 
    given an estimate *I0* for its intensity.

    args
    ----
        I       :   2D ndarray, the PSF image
        I0      :   float, intensity estimate

    returns
    -------
        float, SNR estimate

    """
    return stable_divide_float(
        amp_from_I(I0)**2, ring_var(I, ddof=1),
        inf=np.inf)

def invert_hessian(H, ridge=0.0001):
    """
    Invert a Hessian with ridge regularization to 
    stabilize the inversion.

    args
    ----
        H       :   2D ndarray, shape (n, n)
        ridge   :   float, regularization term

    returns
    -------
        2D ndarray, shape (n, n), the 
            inverted Hessian

    """
    D = np.diag(np.ones(H.shape[0])*ridge)
    while 1:
        try:
            H_inv = np.linalg.inv(H-D)
            return H_inv 
        except (ZeroDivisionError, np.linalg.linalg.LinAlgError):
            D *= 10 
            continue 

def check_2d_gauss_fit(img_shape, pars, max_I0=10000):
    """
    Check whether the fit parameters for a 2D symmetric 
    Gaussian with static sigma are sane.

    This includes:
        - Are the y and x coordinates inside the PSF
            window?
        - Are there negative intensities?
        - Are there crazily high intensities?

    args
    ----
        img_shape       :   (int, int), the PSF image shape
        pars            :   1D ndarray, (y, x, I0, bg), the
                            fit parameters
        max_I0          :   float, maximum tolerated intensity

    returns
    -------
        bool, True if the fit passes the checks

    """
    return all([
        pars[0]>=0,
        pars[1]>=0,
        pars[0]<img_shape[0],
        pars[1]<img_shape[1],
        pars[2]>=0,
        pars[2]<=max_I0,
        pars[3]>=0
    ])

## PSF DEFINITIONS

@lru_cache(maxsize=1)
def psf_int_proj_denom(sigma):
    """
    Convenience function to produce the denominator
    for the Gaussian functions in int_gauss_psf_1d().

    args
    ----
        sigma   :   float

    returns
    -------
        float, np.sqrt(2 * (sigma**2))

    """
    return np.sqrt(2*(sigma**2))


@lru_cache(maxsize=1)
def indices(size_y, size_x):
    """
    Convenience wrapper for np.indices.

    args
    ----
        size_y  :   int, the number of indices
                    in the y direction
        size_x  :   int, the number of indices
                    in the x direction

    returns
    -------
        (
            2D ndarray, the Y indices of each pixel;
            2D ndarray, the X indices of each pixel
        )

    """
    return np.indices((size_y, size_x))

@lru_cache(maxsize=1)
def indices_1d(size_y, size_x):
    """
    Cached convenience function to generate two
    sets of 1D indices.

    args
    ----
        size_y      :   int, the number of indices
                        in the y direction
        size_x      ;   int, the number of indices
                        in the x direction

    returns
    -------
        (
            1D ndarray, the Y indices of each pixel;
            1D ndarray, the X indices of each pixel
        )

    """
    return np.arange(size_y).astype('float64'), \
        np.arange(size_x).astype('float64')

def int_gauss_psf_1d(Y, yc, sigma=1.0):
    """
    Return a 2D integrated Gaussian PSF with unit 
    intensity, projected onto one of the axes. 

    args
    ----
        Y       :   1D ndarray, coordinates along the axis
        yc      :   float, PSF center
        sigma   :   float, Gaussian sigma

    returns
    -------
        1D ndarray, the intensities projected along
            each row of pixels

    """
    S = psf_int_proj_denom(sigma)
    return 0.5*(erf((Y+0.5-yc)/S) - \
        erf((Y-0.5-yc)/S))

def int_gauss_psf_2d(size_y, size_x, yc, xc, I0, sigma=1.0):
    """
    Return a 2D integrated Gaussian PSF with intensity I0.
    Does not include a background term.

    args
    ----
        size_y  :   int, the number of pixels in the y
                    direction
        size_x  :   int, the number of pixels in the x
                    direction
        yc      :   float, center of PSF in y
        xc      :   float, center of PSF in x
        sigma   :   float, PSF width 

    returns
    -------
        2D ndarray of shape (size_y, size_x), the 
            PSF 

    """
    Y, X = indices_1d(size_y, size_x)
    return I0*np.outer(
        int_gauss_psf_1d(Y, yc, sigma=sigma),
        int_gauss_psf_1d(X, xc, sigma=sigma),
    )

def int_gauss_psf_deriv_1d(Y, yc, sigma=1.0):
    """
    Evaluate the derivative of an integrated Gaussian
    PSF model with unit intensity projected onto 1 axis
    with respect to its axis variable.

    args
    ----
        Y       :   1D ndarray, the pixel indices at 
                    which to evaluate the PSF 
        yc      :   float, the spot center
        sigma   :   float, Gaussian sigma

    returns
    -------
        1D ndarray, the derivative of the projection
            with respect to the axis variable at 
            each pixel

    """
    A, B = psf_point_proj_denom(sigma)
    return (np.exp(-(Y-0.5-yc)**2 / A) - \
        np.exp(-(Y+0.5-yc)**2 / A)) / B

@lru_cache(maxsize=1)
def psf_point_proj_denom(sigma):
    """
    Convenience function to produce the denominator
    for the Gaussian functions in psf_point_psf_1d().

    args
    ----
        sigma   :   float, Gaussian sigma

    returns
    -------
        float, float

    """
    S = 2*(sigma**2)
    return S, np.sqrt(np.pi*S)

def point_gauss_psf_1d(Y, yc, sigma=1.0):
    """
    Evaluate a 1D point Gaussian with unit intensity
    at a set of points.

    args
    ----
        Y       :   1D ndarray, coordinates along the axis
        yc      :   float, PSF center
        sigma   :   float, Gaussian sigma

    returns
    -------
        1D ndarray, the intensities projected along
            each row of pixels

    """
    A, B = psf_point_proj_denom(sigma)
    return np.exp(-(Y-yc)**2 / A) / B 

def point_gauss_psf_2d(size_y, size_x, yc, xc, 
    I0, sigma=1.0):
    """
    Return a 2D pointwise-evaluated Gaussian PSF
    with intensity I0.

    args
    ----
        size_y      :   int, size of the PSF subwindow
                        in y 
        size_x      :   int, size of the PSF subwindow
                        in x
        yc          :   float, PSF center in y 
        xc          :   float, PSF center in x
        I0          :   float, PSF intensity 
        sigma       :   float, Gaussian sigma

    returns
    -------
        2D ndarray, the PSF model

    """
    Y, X = indices_1d(size_y, size_x)
    return I0*np.outer(
        point_gauss_psf_1d(Y, yc, sigma=sigma),
        point_gauss_psf_1d(X, xc, sigma=sigma),
    )

## CORE FITTING ROUTINES

def rs(psf_image):
    """
    Localize the center of a PSF using the radial 
    symmetry method.

    Originally conceived by the criminally underrated
    Parasarathy R Nature Methods 9, pgs 724–726 (2012).

    args
    ----
        psf_image : 2D ndarray, PSF subwindow

    returns
    -------
        float y estimate, float x estimate

    """
    # Get the size of the image frame and build
    # a set of pixel indices to match
    N, M = psf_image.shape
    N_half = N // 2
    M_half = M // 2
    ym, xm = np.mgrid[:N-1, :M-1]
    ym = ym - N_half + 0.5
    xm = xm - M_half + 0.5 
    
    # Calculate the diagonal gradients of intensities across each
    # corner of 4 pixels
    dI_du = psf_image[:N-1, 1:] - psf_image[1:, :M-1]
    dI_dv = psf_image[:N-1, :M-1] - psf_image[1:, 1:]
    
    # Smooth the image to reduce the effect of noise, at the cost
    # of a little resolution
    fdu = ndi.uniform_filter(dI_du, 3)
    fdv = ndi.uniform_filter(dI_dv, 3)
    
    dI2 = (fdu ** 2) + (fdv ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = -(fdv + fdu) / (fdu - fdv)
        
    # For pixel values that blow up, instead set them to a very
    # high float
    m[np.isinf(m)] = 9e9
    
    b = ym - m * xm

    sdI2 = dI2.sum()
    ycentroid = (dI2 * ym).sum() / sdI2
    xcentroid = (dI2 * xm).sum() / sdI2
    w = dI2 / np.sqrt((xm - xcentroid)**2 + (ym - ycentroid)**2)

    # Correct nan / inf values
    w[np.isnan(m)] = 0
    b[np.isnan(m)] = 0
    m[np.isnan(m)] = 0

    # Least-squares analytical solution to the point of 
    # maximum radial symmetry, given the slopes at each
    # edge of 4 pixels
    wm2p1 = w / ((m**2) + 1)
    sw = wm2p1.sum()
    smmw = ((m**2) * wm2p1).sum()
    smw = (m * wm2p1).sum()
    smbw = (m * b * wm2p1).sum()
    sbw = (b * wm2p1).sum()
    det = (smw ** 2) - (smmw * sw)
    xc = (smbw*sw - smw*sbw)/det
    yc = (smbw*smw - smmw*sbw)/det

    # Adjust coordinates so that they're relative to the
    # edge of the image frame
    yc = (yc + (N + 1) / 2.0) - 1
    xc = (xc + (M + 1) / 2.0) - 1

    return yc, xc

def fit_ls_int_gaussian(img, guess, sigma=1.0, ridge=0.0001, 
    max_iter=20, damp=0.3, convergence=1.0e-4, divergence=1.0):
    """
    Given an observed spot, estimate the maximum likelihood
    parameters for a 2D integrated Gaussian PSF sampled with
    normally-distributed noise. Here, we use a Levenberg-
    Marquardt procedure to find the ML parameters.

    The model parameters are (y, x, I0, bg), where y, x
    is the spot center, I0 is the integrated intensity, and
    bg is the average background per pixel.

    The function also returns several parameters that are 
    useful for quality control on the fits.

    args
    ----
        img     :   2D ndarray, the observed PSF 
        guess   :   1D ndarray of shape (4), the initial
                    guess for y, x, I0, and bg
        sigma   :   float, Gaussian PSF width 
        ridge   :   float, initial regularization term
                    for inversion of the Hessian
        max_iter:   int, the maximum number of iterations
                    tolerated
        damp    :   float, damping factor for update vector.
                    Larger means faster convergence but
                    also more unstable.
        convergence :   float, the criterion for fit
                    convergence. Only y and x are tested
                    for convergence.
        divergence  :   float, the criterion for fit
                    divergence. Fitting is terminated
                    when this is reached. CURRENTLY NOT
                    IMPLEMENTED.

    returns
    -------
        (
            1D ndarray, the final parameter estimate;

            1D ndarray, the estimated error in each
                parameter (square root of the inverse
                Hessian diagonal);

            float, the Hessian determinant;

            float, the root mean squared deviation of the
                final PSF model from the observed PSF;

            int, the number of iterations
        )

    """
    n_pixels = img.shape[0] * img.shape[1]

    # 1D indices for each pixel
    index_y, index_x = indices_1d(*img.shape)

    # Update vector
    update = np.ones(4, dtype='float64')

    # Current parameter set
    pars = guess.copy()

    # Continue improving guess until convergence
    iter_idx = 0
    while iter_idx<max_iter and (np.abs(update[:2])>convergence).any():

        # Evaluate the PSF model and its derivatives projected
        # onto each of the two axes
        proj_y = int_gauss_psf_1d(index_y, pars[0], sigma=sigma)
        proj_x = int_gauss_psf_1d(index_x, pars[1], sigma=sigma)

        # Evaluate the derivatives of the model with respect to
        # the Y and X axies
        dproj_y_dy = int_gauss_psf_deriv_1d(index_y, pars[0], sigma=sigma)
        dproj_x_dx = int_gauss_psf_deriv_1d(index_x, pars[1], sigma=sigma)

        # Evaluate the model Jacobian
        J = np.asarray([
            (pars[2] * np.outer(dproj_y_dy, proj_x)).ravel(),
            (pars[2] * np.outer(proj_y, dproj_x_dx)).ravel(),
            (np.outer(proj_y, proj_x)).ravel(),
            np.ones(n_pixels)
        ]).T 

        # Evaluate the model 
        model = pars[2] * J[:,2] + pars[3]

        # Estimate the noise variance
        devs = img.ravel() - model 
        sig2 = (devs**2).mean()

        # Get the gradient of the log-likelihood
        grad = (J.T * devs).sum(axis=1) / sig2 

        # Evaluate and invert the Hessian
        H_inv = invert_hessian(-J.T@J/sig2, ridge=ridge)

        # Get the update vector
        update[:] = -H_inv @ grad

        # Apply the update to parameter estimate 
        pars = pars + damp * update 
        iter_idx += 1

    # Estimate the error with the inverse Hessian
    err = np.sqrt(np.diagonal(-H_inv))

    # Evaluate the Hessian determinant, which is 
    # also useful as a metric of error
    return pars, err, np.linalg.det(H_inv), np.sqrt(sig2), iter_idx

def fit_ls_point_gaussian(img, guess, sigma=1.0, ridge=0.0001, 
    max_iter=20, damp=0.3, convergence=1.0e-4, divergence=1.0):
    """
    Given an observed spot, estimate the maximum likelihood
    parameters for a 2D pointwise-evaluated Gaussian PSF 
    sampled with normally-distributed noise. This function
    uses a Levenberg-Marquardt procedure to find the ML 
    parameters.

    The underlying model for the pointwise-evaluated and 
    integrated Gaussian PSFs is identical - a symmetric 2D
    Gaussian. The distinction is how they handle sampling
    on discrete pixels:

        - the point Gaussian takes the value on each pixel
            to be equal to the Gaussian evaluated at the 
            center of the pixel

        - the integrated Gaussian takes the value on each 
            pixel to be equal to the Gaussian integrated
            across the whole area of that pixel

    Integrated Gaussians are more accurate and less prone
    to edge biases, but the math is slightly more 
    complicated and fitting is slower as a result.

    The model parameters are (y, x, I0, bg), where y, x
    is the spot center, I0 is the integrated intensity, and
    bg is the average background per pixel.

    The function also returns several parameters that are 
    useful for quality control on the fits.

    args
    ----
        img     :   2D ndarray, the observed PSF 
        guess   :   1D ndarray of shape (4), the initial
                    guess for y, x, I0, and bg
        sigma   :   float, Gaussian PSF width 
        ridge   :   float, initial regularization term
                    for inversion of the Hessian
        max_iter:   int, the maximum number of iterations
                    tolerated
        damp    :   float, damping factor for update vector.
                    Larger means faster convergence but
                    also more unstable.
        convergence :   float, the criterion for fit
                    convergence. Only y and x are tested
                    for convergence.
        divergence  :   float, the criterion for fit
                    divergence. Fitting is terminated
                    when this is reached. CURRENTLY NOT
                    IMPLEMENTED.

    returns
    -------
        (
            1D ndarray, the final parameter estimate;

            1D ndarray, the estimated error in each
                parameter (square root of the inverse
                Hessian diagonal);

            float, the Hessian determinant;

            float, the root mean squared deviation of the
                final PSF model from the observed PSF;

            int, the number of iterations
        )

    """
    S2 = sigma**2
    n_pixels = img.shape[0] * img.shape[1]

    # 1D indices for each pixel
    index_y, index_x = indices_1d(*img.shape)

    # Update vector
    update = np.ones(4, dtype='float64')

    # Current parameter set
    pars = guess.copy()

    # Continue improving guess until convergence
    iter_idx = 0
    while iter_idx<max_iter and (np.abs(update[:2])>convergence).any():

        # Evaluate a unit intensity Gaussian projected onto each
        # axis
        psf_1d_y = point_gauss_psf_1d(index_y, pars[0], sigma=sigma)
        psf_1d_x = point_gauss_psf_1d(index_x, pars[1], sigma=sigma)

        # Evaluate the derivatives of the model with respect to
        # the Y and X axes
        dpsf_1d_y_dy = (index_y-pars[0]) * psf_1d_y / S2
        dpsf_1d_x_dx = (index_x-pars[1]) * psf_1d_x / S2 

        # Evaluate the model Jacobian
        J = np.asarray([
            (pars[2] * np.outer(dpsf_1d_y_dy, psf_1d_x)).ravel(),
            (pars[2] * np.outer(psf_1d_y, dpsf_1d_x_dx)).ravel(),
            (np.outer(psf_1d_y, psf_1d_x)).ravel(),
            np.ones(n_pixels)
        ]).T 

        # Evaluate the model 
        model = pars[2] * J[:,2] + pars[3]

        # Estimate the noise variance
        devs = img.ravel() - model 
        sig2 = (devs**2).mean()

        # Get the gradient of the log-likelihood under a
        # normally-distributed noise model
        grad = (J.T * devs).sum(axis=1) / sig2 

        # Evaluate and invert the Hessian
        H_inv = invert_hessian(-J.T@J/sig2, ridge=ridge)

        # Get the update vector
        update[:] = -H_inv @ grad

        # Apply the update to parameter estimate 
        pars = pars + damp * update 
        iter_idx += 1

    # Estimate the error with the inverse Hessian
    err = np.sqrt(np.diagonal(-H_inv))

    # Evaluate the Hessian determinant, which is 
    # also useful as a metric of error
    return pars, err, np.linalg.det(H_inv), np.sqrt(sig2), iter_idx

def fit_poisson_int_gaussian(img, guess, sigma=1.0, ridge=0.0001, 
    max_iter=20, damp=0.3, convergence=1.0e-4, divergence=1.0):
    """
    Given an observed spot, estimate the maximum likelihood
    parameters for a 2D integrated Gaussian PSF model sampled
    with Poisson noise, using a Levenberg-Marquardt procedure.

    While LM with Poisson noise is a little slower than LS
    routines, it is the most accurate model for the noise on 
    EMCCD cameras.

    The model parameters are (y, x, I0, bg), where y, x
    is the spot center, I0 is the integrated intensity, and
    bg is the average background per pixel.

    The function also returns several parameters that are 
    useful for quality control on the fits.

    args
    ----
        img     :   2D ndarray, the observed PSF 
        guess   :   1D ndarray of shape (4), the initial
                    guess for y, x, I0, and bg
        sigma   :   float, Gaussian PSF width 
        ridge   :   float, initial regularization term
                    for inversion of the Hessian
        max_iter:   int, the maximum number of iterations
                    tolerated
        damp    :   float, damping factor for update vector.
                    Larger means faster convergence but
                    also more unstable.
        convergence :   float, the criterion for fit
                    convergence. Only y and x are tested
                    for convergence.
        divergence  :   float, the criterion for fit
                    divergence. Fitting is terminated
                    when this is reached. CURRENTLY NOT
                    IMPLEMENTED.

    returns
    -------
        (
            1D ndarray, the final parameter estimate;

            1D ndarray, the estimated error in each
                parameter (square root of the inverse
                Hessian diagonal);

            float, the Hessian determinant;

            float, the root mean squared deviation of the
                final PSF model from the observed PSF;

            int, the number of iterations
        )

    """
    n_pixels = img.shape[0] * img.shape[1]
    img_ravel = img.ravel()

    # 1D indices for each pixel
    index_y, index_x = indices_1d(*img.shape)

    # Update vector
    update = np.ones(4, dtype='float64')

    # Current parameter set
    pars = guess.copy()

    # Instantiate the Hessian
    H = np.empty((4, 4), dtype='float64')

    # Continue improving guess until convergence
    iter_idx = 0
    while iter_idx<max_iter and (np.abs(update[:2])>convergence).any():

        # Evaluate the PSF model and its derivatives projected
        # onto each of the two axes
        proj_y = int_gauss_psf_1d(index_y, pars[0], sigma=sigma)
        proj_x = int_gauss_psf_1d(index_x, pars[1], sigma=sigma)

        # Evaluate the derivatives of the model with respect to
        # the Y and X axies
        dproj_y_dy = int_gauss_psf_deriv_1d(index_y, pars[0], sigma=sigma)
        dproj_x_dx = int_gauss_psf_deriv_1d(index_x, pars[1], sigma=sigma)

        # Evaluate the model Jacobian
        J = np.asarray([
            (pars[2] * np.outer(dproj_y_dy, proj_x)).ravel(),
            (pars[2] * np.outer(proj_y, dproj_x_dx)).ravel(),
            (np.outer(proj_y, proj_x)).ravel(),
            np.ones(n_pixels)
        ]).T 

        # Evaluate the model 
        model = pars[2] * J[:,2] + pars[3]

        # if debug:
        #     print('iter: %d' % iter_idx)
        #     wireframe_overlay(img, model.reshape(img.shape))

        # Get the gradient of the log-likelihood
        grad = (J.T * ((img_ravel/model) - 1.0)).sum(axis=1)

        # Evaluate and invert the Hessian
        H_factor = -(img_ravel / (model**2))
        for i in range(4):
            for j in range(i, 4):
                H[i,j] = (H_factor * J[:,i] * J[:,j]).sum()
                if j > i:
                    H[j,i] = H[i,j]
        H_inv = invert_hessian(H, ridge=ridge)

        # Get the update vector
        update[:] = -H_inv @ grad

        # Apply the update to parameter estimate 
        pars = pars + damp * update 
        iter_idx += 1

    # if debug:
    #     print('final:')
    #     wireframe_overlay(img, model.reshape(img.shape))

    # Estimate the error with the inverse Hessian
    err = np.sqrt(np.diagonal(-H_inv))

    # Get the root mean squared deviation of the final
    # model from the observed spot
    rmse = ((model - img_ravel)**2).mean()

    # Evaluate the Hessian determinant, which is 
    # also useful as a metric of error
    return pars, err, np.linalg.det(H_inv), rmse, iter_idx

########################
## TRACKING UTILITIES ##
########################

def connected_components(semigraph):
    """
    Find independent connected subgraphs in a bipartite graph 
    (aka semigraph) by a floodfill procedure.

    The semigraph is a set of edges betwen two sets of nodes-
    one set corresponding to each y-index, and the other 
    corresponding to each x-index. Edges can only exist between
    a y-node and an x-node, so the semigraph is conveniently
    represented as a binary (0/1 values) 2D matrix.

    The goal of this function is to identify independent 
    subgraphs in the original semigraph. If I start on a y-node
    or x-node for one of these independent subgraphs and walk
    along edges, I can reach any point in that subgraph but no
    points in other subgraphs.

    Each subgraph can itself be represented by a 2D ndarray 
    along with a reference to the specific y-nodes and x-dnoes
    in the original semigraph corresponding to that subgraph.

    For the purposes of tracking, it is also convenient to 
    separate y-nodes and x-nodes that are NOT connected by any
    edges to other nodes. These are returned as separate ndarrays:
    y_without_x and x_without_y.
    
    Parameters
    ----------
        semigraph       :   2D ndarray, with only 0 and 1 values
    
    Returns
    -------
    (
        subgraphs           :   list of 2D ndarray
        subgraph_y_indices  :   list of 1D ndarray, the set of 
                                y-nodes corresponding to each 
                                subgraph
        subgraph_x_indices  :   list of 1D ndarray, the set of 
                                x-nodes corresponding to each 
                                subgraph
        y_without_x         :   1D ndarray, the y-nodes that are 
                                not connected to any x-nodes
        x_without_y         :   1D ndarray, the x-nodes that are 
                                not connected to any y-nodes
    )

    """
    # The set of all y-nodes (corresponding to y-indices in the semigraph)
    y_indices = np.arange(semigraph.shape[0]).astype("int64")

    # The set of all x-nodes (corresponding to x-indices in the semigraph)
    x_indices = np.arange(semigraph.shape[1]).astype("int64")

    # Find y-nodes that don't connect to any x-node,
    # and vice versa
    where_y_without_x = semigraph.sum(axis=1) == 0
    where_x_without_y = semigraph.sum(axis=0) == 0
    y_without_x = y_indices[where_y_without_x]
    x_without_y = x_indices[where_x_without_y]

    # Consider the remaining nodes, which have at least one edge
    # to a node of the other class
    semigraph = semigraph[~where_y_without_x, :]
    semigraph = semigraph[:, ~where_x_without_y]
    y_indices = y_indices[~where_y_without_x]
    x_indices = x_indices[~where_x_without_y]

    # For the remaining nodes, keep track of (1) the subgraphs
    # encoding connected components, (2) the set of original y-indices
    # corresponding to each subgraph, and (3) the set of original x-
    # indices corresponding to each subgraph
    subgraphs = []
    subgraph_y_indices = []
    subgraph_x_indices = []

    # Work by iteratively removing independent subgraphs from the
    # graph. The list of nodes still remaining are kept in
    # *unassigned_y* and *unassigned_x*
    unassigned_y, unassigned_x = (semigraph == 1).nonzero()

    # The current index is used to floodfill the graph with that
    # integer. It is incremented as we find more independent subgraphs.
    current_idx = 2

    # While we still have unassigned nodes
    while len(unassigned_y) > 0:

        # Start the floodfill somewhere with an unassigned y-node
        semigraph[unassigned_y[0], unassigned_x[0]] = current_idx

        # Keep going until subsequent steps of the floodfill don't
        # pick up additional nodes
        prev_nodes = 0
        curr_nodes = 1
        while curr_nodes != prev_nodes:
            # Only floodfill along existing edges in the graph
            where_y, where_x = (semigraph == current_idx).nonzero()

            # Assign connected nodes to the same subgraph index
            semigraph[where_y, :] *= current_idx
            semigraph[semigraph > current_idx] = current_idx
            semigraph[:, where_x] *= current_idx
            semigraph[semigraph > current_idx] = current_idx

            # Correct for re-finding the same nodes and multiplying
            # them more than once (implemented in the line above)
            # semigraph[semigraph > current_idx] = current_idx

            # Update the node counts in this subgraph
            prev_nodes = curr_nodes
            curr_nodes = (semigraph == current_idx).sum()
        current_idx += 1

        # Get the local indices of the y-nodes and x-nodes (in the context
        # of the remaining graph)
        where_y = np.unique(where_y)
        where_x = np.unique(where_x)

        # Use the local indices to pull this subgraph out of the
        # main graph
        subgraph = semigraph[where_y, :]
        subgraph = subgraph[:, where_x]

        # Save the subgraph
        if not (subgraph.shape[0] == 0 and subgraph.shape[0] == 0):
            subgraphs.append(subgraph)

            # Get the original y-nodes and x-nodes that were used in this
            # subgraph
            subgraph_y_indices.append(y_indices[where_y])
            subgraph_x_indices.append(x_indices[where_x])

        # Update the list of unassigned y- and x-nodes
        unassigned_y, unassigned_x = (semigraph == 1).nonzero()

    return subgraphs, subgraph_y_indices, subgraph_x_indices, y_without_x, x_without_y

def concat_tracks(*tracks):
    """
    Join some trajectory dataframes together into a larger dataframe,
    while preserving uniqe trajectory indices.

    args
    ----
        tracks      :   pandas.DataFrame with the "trajectory" column

    returns
    -------
        pandas.DataFrame, the concatenated trajectories

    """
    n = len(tracks)

    # Sort the tracks dataframes by their size. The only important thing
    # here is that if at least one of the tracks dataframes is nonempty,
    # we need to put that one first.
    df_lens = [len(t) for t in tracks]
    try:
        tracks = [t for _, t in sorted(zip(df_lens, tracks))][::-1]
    except ValueError:
        pass

    # Iteratively concatenate each dataframe to the first while 
    # incrementing the trajectory index as necessary
    out = tracks[0].assign(dataframe_index=0)
    c_idx = out["trajectory"].max() + 1

    for t in range(1, n):

        # Get the next set of trajectories and keep track of the origin
        # dataframe
        new = tracks[t].assign(dataframe_index=t)

        # Ignore negative trajectory indices (facilitating a user filter)
        new.loc[new["trajectory"]>=0, "trajectory"] += c_idx 

        # Increment the total number of trajectories
        c_idx = new["trajectory"].max() + 1

        # Concatenate
        out = pd.concat([out, new], ignore_index=True, sort=False)

    return out 
    
def track_length(tracks):
    """
    Generate a new column with the trajectory length in frames.

    args
    ----
        tracks  :   pandas.DataFrame

    returns
    -------
        pandas.DataFrame, the input dataframe with a new 
            column "track_length"

    """
    if "track_length" in tracks.columns:
        tracks = tracks.drop("track_length", axis=1)
    tracks = tracks.join(
        tracks.groupby("trajectory").size().rename("track_length"),
        on="trajectory"
    )
    return tracks 

#######################
## MASKING UTILITIES ##
#######################

def get_edges(bin_img):
    """
    Given a binary image that is False outside of an object and True
    inside of it, return another binary image that is True for points
    in the original image that border a False pixel, and False otherwise.

    args
    ----
        bin_img         :   2D ndarray, dtype bool

    returns
    -------
        2D ndarray, dtype bool, shape shape as bin_img

    """
    assert bin_img.dtype == np.bool, "get_edges: must be bool input"
    out = np.zeros(bin_img.shape, dtype="bool")

    # Find the edges of the binary mask
    out[1:, :] = out[1:, :] | (bin_img[1:, :] & ~bin_img[:-1, :])
    out[:, 1:] = out[:, 1:] | (bin_img[:, 1:] & ~bin_img[:, :-1])
    out[:-1, :] = out[:-1, :] | (bin_img[:-1, :] & ~bin_img[1:, :])
    out[:, :-1] = out[:, :-1] | (bin_img[:, :-1] & ~bin_img[:, 1:])

    # Figure out where the object in the binary image intersects
    # the edge of the image frame
    out[0, :] = out[0, :] | bin_img[0, :]
    out[:, 0] = out[:, 0] | bin_img[:, 0]
    out[-1, :] = out[-1, :] | bin_img[-1, :]
    out[:, -1] = out[:, -1] | bin_img[:, -1]

    return out

def get_ordered_mask_points(mask, max_points=100):
    """
    Given the edges of a two-dimensional binary mask, construct a line
    around the mask.

    args
    ----
        mask        :   2D ndarray, dtype bool, mask edges as 
                        returned by get_edges
        max_points  :   int, the maximum number of points tolerated
                        in the final mask. If the number of points 
                        exceeds this, the points are repeatedly 
                        downsampled until there are fewer than 
                        max_points.

    returns
    -------
        2D ndarray of shape (n_points, 2), the points belonging
            to this ROI

    """
    # Get the X and Y coordinates of all points in the mask edge
    points = np.asarray(mask.nonzero()).T

    # Keep track of which points we've included so far
    included = np.zeros(points.shape[0], dtype=np.bool)

    # Start at the first point
    ordered_points = np.zeros(points.shape, dtype=points.dtype)
    ordered_points[0,:] = points[0,:]
    included[0] = True 

    # Index of the current point
    c = 0
    midx = 0

    # Find the closest point to the current point
    while c < points.shape[0]-1:

        # Compute distances to every other point
        distances = np.sqrt(((points[midx,:]-points)**2).sum(axis=1))

        # Set included points to impossible distances
        distances[included] = np.inf 

        # Among the points not yet included in *ordered_points*,
        # choose the one closest to the current point
        midx = np.argmin(distances)

        # Add this point to the set of ordered points
        ordered_points[c+1,:] = points[midx, :]

        # Mark this point as included
        included[midx] = True 

        # Increment the current point counter
        c += 1

    # Downsample until there are fewer than max_points
    while ordered_points.shape[0] > max_points:
        ordered_points = ordered_points[::2,:]

    return ordered_points


