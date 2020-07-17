#!/usr/bin/env python
"""
subpixel.py -- localize PSFs to subpixel resolution

"""
# Numeric
import numpy as np 

# Dataframes
import pandas as pd 

# Image processing 
from scipy import ndimage as ndi 

# Low-level subpixel localization utilities
from .helper import (
    assign_methods,
    ring_mean, 
    rs,
    I0_is_crazy,
    estimate_I0,
    estimate_I0_multiple_points,
    estimate_snr,
    fit_ls_int_gaussian,
    fit_ls_point_gaussian,
    fit_poisson_int_gaussian,
    check_2d_gauss_fit
)

def centroid(I, sub_bg=False):
    """
    Find the center of a spot by taking its center
    of mass.

    args
    ----
        I       :   2D ndarray (YX), spot image
        sub_bg  :   bool, subtract background prior to 
                    taking center of mass

    returns
    -------
        dict {
            y           : y centroid (pixels),
            x           : x centroid (pixels), 
            bg          : estimated background intensity per 
                            pixel (AU),
            I0          : integrated spot intensity above
                            background (AU),
            error_flag  : int, the error code. 0 indicates
                            no errors.
        }

    """
    # Estimate background
    bg = ring_mean(I)

    # Subtract background
    I_sub = np.clip(I-bg, 0.0, np.inf)

    # Integrated intensity above background
    I0 = I_sub.sum()

    # Find spot centers
    if sub_bg:
        y, x = ndi.center_of_mass(I_sub)
    else:
        y, x = ndi.center_of_mass(I)

    # Estimate SNR
    snr = estimate_snr(I, I0)

    # Return parameter estimates 
    return dict((
        ('y', y),
        ('x', x),
        ('I0', I0),
        ('bg', bg),
        ('error_flag', 0),
        ('snr', snr)
    ))

def radial_symmetry(I, sigma=1.0):
    """
    Estimate the center of a spot by the radial symmetry
    method, described in Parthasarathy et al. Nat Met 2012.

    Also infer the intensity of the spots assuming an 
    integrated Gaussian PSF. This is useful as a first guess
    for iterative localization techniques.

    args
    ----
        I       :   2D ndarray, spot image
        sigma   :   float, static integrated 2D Gaussian 
                    width

    returns
    -------
        dict {
            y   :    estimated y center (pixels),
            x   :    estimated x center (pixels),
            I0  :    estimated PSF intensity (AU),
            bg  :    estimated background intensity per
                        pixel (AU),
            error_flag  :   int, the error flag. 0
                            if there are no errors,
        }

    """
    # Estimate spot centers
    y, x = rs(I)

    # Estimate background level
    bg = ring_mean(I)

    # Estimate Gaussian intensity
    I0 = estimate_I0_multiple_points(I, y, 
        x, bg, sigma=sigma)

    # If I0 is crazy, use a different guess
    if I0_is_crazy(I0):
        I0 = np.clip(I-bg, 0.0, np.inf).sum()
        error_flag = 1
    else:
        error_flag = 0

    # Estimate SNR
    snr = estimate_snr(I, I0)

    # Return parameter estimate
    return dict((
        ('y', y),
        ('x', x),
        ('I0', I0),
        ('bg', bg),
        ('error_flag', error_flag),
        ('snr', snr)
    ))

def ls_int_gaussian(I, sigma=1.0, ridge=0.0001, max_iter=10,
    damp=0.3, convergence=1.0e-4, divergence=1.0):
    """
    Estimate the maximum likelihood parameters for a symmetric
    2D integrated Gaussian PSF model, given an observed spot 
    *I* with normally-distributed noise.

    This method uses radial symmetry for a first guess, followed
    by a Levenberg-Marquardt routine for refinement. The core
    calculation is performed in quot.helper.fit_ls_int_gaussian.

    args
    ----
        I           :   2D ndarray (YX), the observed spot
        sigma       :   float, static Gaussian width 
        ridge       :   float, initial regularization term
                        for inversion of the Hessian
        max_iter    :   int, the maximum tolerated number
                        of iterations
        damp        :   float, damping factor for the update
                        at each iteration. Larger means faster
                        convergence, but less stable.
        convergence :   float, maximum magnitude of the update
                        vector at which to call convergence. 
        divergence  :   float, divergence criterion

    returns
    -------
        dict, the parameter estimate, error estimates, and 
            related parameters about the fitting problem 
            (see below). Some of these can be useful for 
            QC.

    """
    # Make the initial guess by radial symmetry
    guess = np.array([*rs(I), 0.0, ring_mean(I)])
    guess[2] = estimate_I0(I, guess[0], guess[1], guess[3],
        sigma=sigma)

    # Run the fitting routine
    pars, err, H_det, rmse, n_iter = fit_ls_int_gaussian(
        I, guess, sigma=sigma, ridge=ridge, max_iter=max_iter,
        damp=damp, convergence=convergence, divergence=divergence)

    # Check for crazy fits
    if check_2d_gauss_fit(I.shape, pars):
        error_flag = 0
    else:
        error_flag = 1

    # Estimate SNR
    snr = estimate_snr(I, pars[2])

    # Return parameter estimate
    return dict((
        ('y', pars[0]),
        ('x', pars[1]),
        ('I0', pars[2]),
        ('bg', pars[3]),
        ('y_err', err[0]),
        ('x_err', err[1]),
        ('I0_err', err[2]),
        ('bg_err', err[3]),
        ('H_det', H_det),
        ('error_flag', error_flag),
        ('snr', snr),       
        ('rmse', rmse),
        ('n_iter', n_iter)
    ))

def ls_point_gaussian(I, sigma=1.0, ridge=0.0001, max_iter=10,
    damp=0.3, convergence=1.0e-4, divergence=1.0):
    """
    Estimate the maximum likelihood parameters for a symmetric
    2D pointwise-evaluated Gaussian PSF model, given an observed
    spot *I* with normally-distributed noise.

    Both integrated and pointwise-evaluated models have the same
    underlying model: a symmetric, 2D Gaussian PSF. The 
    distinction is how they handle sampling on discrete pixels:

        - the point Gaussian takes the value on each pixel to 
            be equal to the PSF function evaluated at the center
            of that pixel

        - the integrated Gaussian takes the value on each pixel
            to be equal to the PSF integrated across the whole
            area of the pixel

    Integrated Gaussian models are more accurate and less prone
    to edge biases, at the cost of increased complexity and 
    perhaps lower speed.

    This method uses radial symmetry for a first guess, followed
    by a Levenberg-Marquardt routine for refinement. The core
    calculation is performed in quot.helper.fit_ls_point_gaussian.

    args
    ----
        I           :   2D ndarray (YX), the observed spot
        sigma       :   float, static Gaussian width 
        ridge       :   float, initial regularization term
                        for inversion of the Hessian
        max_iter    :   int, the maximum tolerated number
                        of iterations
        damp        :   float, damping factor for the update
                        at each iteration. Larger means faster
                        convergence, but less stable.
        convergence :   float, maximum magnitude of the update
                        vector at which to call convergence. 
        divergence  :   float, divergence criterion

    returns
    -------
        dict, the parameter estimate, error estimates, and 
            related parameters about the fitting problem 
            (see below). Some of these can be useful for 
            QC.

    """
    # Make the initial guess by radial symmetry
    guess = np.array([*rs(I), 0.0, ring_mean(I)])
    guess[2] = estimate_I0(I, guess[0], guess[1], guess[3],
        sigma=sigma)

    # Run the fitting routine
    pars, err, H_det, rmse, n_iter = fit_ls_point_gaussian(
        I, guess, sigma=sigma, ridge=ridge, max_iter=max_iter,
        damp=damp, convergence=convergence, divergence=divergence)

    # Check for crazy fits
    if check_2d_gauss_fit(I.shape, pars):
        error_flag = 0
    else:
        error_flag = 1

    # Estimate SNR
    snr = estimate_snr(I, pars[2])

    # Return parameter estimate
    return dict((
        ('y', pars[0]),
        ('x', pars[1]),
        ('I0', pars[2]),
        ('bg', pars[3]),
        ('y_err', err[0]),
        ('x_err', err[1]),
        ('I0_err', err[2]),
        ('bg_err', err[3]),
        ('H_det', H_det),
        ('error_flag', error_flag),
        ('snr', snr),       
        ('rmse', rmse),
        ('n_iter', n_iter)
    ))

def poisson_int_gaussian(I, sigma=1.0, ridge=0.0001, max_iter=10,
    damp=0.3, convergence=1.0e-4, divergence=1.0):
    """
    Estimate the maximum likelihood parameters for a symmetric
    2D integrated Gaussian PSF model, given an observed spot 
    *I* with Poisson-distributed noise.

    This method uses radial symmetry for a first guess, followed
    by a Levenberg-Marquardt routine for refinement. The core
    calculation is performed in quot.helper.fit_poisson_int_gaussian.

    args
    ----
        I           :   2D ndarray (YX), the observed spot
        sigma       :   float, static Gaussian width 
        ridge       :   float, initial regularization term
                        for inversion of the Hessian
        max_iter    :   int, the maximum tolerated number
                        of iterations
        damp        :   float, damping factor for the update
                        at each iteration. Larger means faster
                        convergence, but less stable.
        convergence :   float, maximum magnitude of the update
                        vector at which to call convergence. 
        divergence  :   float, divergence criterion

    returns
    -------
        dict, the parameter estimate, error estimates, and 
            related parameters about the fitting problem 
            (see below). Some of these can be useful for 
            QC.

    """
    # Make the initial guess by radial symmetry
    guess = np.array([*rs(I), 0.0, ring_mean(I)])
    guess[2] = estimate_I0(I, guess[0], guess[1], guess[3],
        sigma=sigma)

    # Run the fitting routine
    pars, err, H_det, rmse, n_iter = fit_poisson_int_gaussian(
        I, guess, sigma=sigma, ridge=ridge, max_iter=max_iter,
        damp=damp, convergence=convergence, divergence=divergence)

    # Check for crazy fits
    if check_2d_gauss_fit(I.shape, pars):
        error_flag = 0
    else:
        error_flag = 1

    # Estimate SNR
    snr = estimate_snr(I, pars[2])

    # Return parameter estimate
    return dict((
        ('y', pars[0]),
        ('x', pars[1]),
        ('I0', pars[2]),
        ('bg', pars[3]),
        ('y_err', err[0]),
        ('x_err', err[1]),
        ('I0_err', err[2]),
        ('bg_err', err[3]),
        ('H_det', H_det),
        ('error_flag', error_flag),
        ('snr', snr),       
        ('rmse', rmse),
        ('n_iter', n_iter)
    ))

##########################################
## MAIN SUBPIXEL LOCALIZATION FUNCTIONS ##
##########################################

# All localization methods available
METHODS = {
    'centroid': centroid,
    'radial_symmetry': radial_symmetry,
    'ls_int_gaussian': ls_int_gaussian,
    'ls_point_gaussian': ls_point_gaussian,
    'poisson_int_gaussian': poisson_int_gaussian
}

# Wrapper for all localization methods to run on 
# a single PSF. 

# Example usage:
#  fit_pars = localize(psf_img, method='ls_int_gaussian', 
#     **method_kwargs)
localize = assign_methods(METHODS)(lambda I, **kwargs: None)

# Wrapper for all localization methods on a frame
# with several detections.
def localize_frame(img, positions, method=None, window_size=9,
    camera_bg=0.0, camera_gain=1.0, **method_kwargs):
    """
    Run localization on multiple spots in a large 2D image,
    returning the result as a pandas DataFrame.

    args
    ----
        img         :   2D ndarray (YX), the image frame
        positions   :   2D ndarray of shape (n_spots, 2),
                        the y and x positions at which to 
                        localize spots
        method      :   str, a method in METHODS
        window_size :   int, the fitting window size
        camera_bg   :   float, the BG per subpixel in the camera
        camera_gain :   float, the camera gain (grayvalues/photon)
        method_kwargs:  to the localization method

    returns
    -------
        pandas.DataFrame with columns ['y_detect', 'x_detect']
            plus all of the outputs from the localization function

    """
    # If no method passed, return the unmodified dataframe
    if method is None:
        return positions 

    # Remove detections too close to the edge for a square
    # subwindow
    if len(positions.shape)==2:
        hw = window_size // 2
        positions = positions[
            (positions[:,0]>=hw) & (positions[:,0]<img.shape[0]-hw) & \
            (positions[:,1]>=hw) & (positions[:,1]<img.shape[1]-hw)
        , :]

        # Get the localization method
        method_f = METHODS.get(method)

        # Localize a PSF in a subwindow of the image
        def localize_subwindow(yd, xd):
            psf_img = np.clip(
                img[yd-hw:yd+hw+1, xd-hw:xd+hw+1]-camera_bg, 0, np.inf
            ) / camera_gain 
            r = method_f(psf_img, **method_kwargs)
            r.update({"y_detect": yd, "x_detect": xd})
            r['y'] = r['y'] + yd - hw 
            r['x'] = r['x'] + xd - hw 
            return r

        # Run localization on all PSF subwindows
        result = pd.DataFrame([localize_subwindow(yd, xd) \
            for yd, xd in positions])
        return result 

    else:
        return pd.DataFrame([])

