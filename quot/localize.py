"""
localize.py

"""
# Numerics
import numpy as np 

# Image processing
from scipy import ndimage as ndi 

# Dataframes
import pandas as pd 

# File reader 
from quot import qio 

# Image filtering utilities
from quot import image_filter 

# Detection utilities
from quot import detect 

# General package-specific utilities
from quot import utils 

# Progress bar
from tqdm import tqdm 

def centroid(psf_img, sub_bg=False, camera_offset=0.0,
    camera_gain=1.0):
    """
    Localize with center-of-mass.

    args
    ----
        psf_img : 2D ndarray
        sub_bg : bool, subtract BG before
            taking centroid
        camera_offset : float, offset of 
            camera
        camera_gain : float, gain of 
            camera

    returns
    -------
        dict : {
            'y0': y centroid in pixels,
            'x0': x centroid in pixels 
            'I': integrated intensity above background;
            'amp': the intensity of the brightest pixel;
            'bg': the estimated background level;
            'snr': the estimated signal-to-noise ratio
                based on the amplitude
        }

    """
    # Subtract BG and divide out gain 
    psf_img = utils.rescale_img(psf_img, 
        camera_offset=camera_offset,
        camera_gain=camera_gain)

    # Estimate background
    bg = utils.ring_mean(psf_img)

    # Subtract background
    psf_sub_bg = utils.set_neg_to_zero(psf_img-bg)

    # Integrate intensity above background
    I = psf_sub_bg.sum()

    # Take the brightest pixel above background as 
    # the amplitude
    amp = utils.amp_from_I(I, sigma=1.0)
    
    # Find spot centers
    if sub_bg:
        y0, x0 = ndi.center_of_mass(psf_sub_bg)
    else:
        y0, x0 = ndi.center_of_mass(psf_img)

    # Estimate signal-to-noise
    snr = utils.estimate_snr(psf_img, amp)

    return dict((('bg', bg), ('I', I), ('amp', amp),
        ('y0', y0), ('x0', x0), ('snr', snr)))

def radial_symmetry(psf_img, sigma=1.0, camera_offset=0.0,
    camera_gain=1.0):
    """
    Find spot centers by radial symmetry, and infer
    the intensity of the spots using a Gaussian model.

    args
    ----
        psf_img : 2D ndarray
        sigma : float, Gaussian sigma
        camera_offset : float, offset of 
            camera
        camera_gain : float, gain of 
            camera

    returns
    -------
        dict: {
            'y0': y center,
            'x0': x center,
            'I': estimated integrated intensity of 
                Gaussian,
            'amp': estimated Gaussian peak amplitude;
            'bg': estimated background level;
            'snr': estimated signal to noise ratio
        }

    """
    # Subtract BG and divide out gain 
    psf_img = utils.rescale_img(psf_img, 
        camera_offset=camera_offset,
        camera_gain=camera_gain)

    # Estimate spot centers
    y0, x0 = utils.rs(psf_img)

    # Estimate background level
    bg = utils.ring_mean(psf_img)

    # Estimate the intensity of the Gaussian
    I = utils.estimate_intensity(psf_img, y0, x0,
        bg, sigma=sigma)

    # Estimate peak amplitude of Gaussian
    amp = utils.amp_from_I(I, sigma=sigma)

    # Estimate SNR
    snr = utils.estimate_snr(psf_img, amp)

    # Return dict with values as output
    return dict((('y0', y0), ('x0', x0), ('bg', bg),
        ('I', I), ('amp', amp), ('snr', snr)))

def mle_poisson(psf_img, sigma=1.0, max_iter=20,
    damp=0.3, camera_offset=0.0, camera_gain=1.0,
    convergence_crit=3.0e-5, ridge=1.0e-4, 
    debug=False):
    """
    Estimate the maximum likelihood model 
    parameters for an integrated 2D Gaussian
    PSF under a Poisson noise model, using
    a Levenberg-Marquardt procedure.

    Due to the nature of Poisson noise, it is
    recommended to measure the camera gain and
    offset to convert from grayvalues to photons.

    args
    ----
        psf_img : 2D ndarray
        sigma : float
        max_iter : int
        damp : float
        camera_offset, camera_bg : floats
        convergence_crit : float
        ridge : float, initial regularization
            term
        debug : bool, show each stage
            of fitting

    returns
    -------
        dict : {
            'y0': y center,
            'x0': x center,
            'I': intensity,
            'bg': background intensity,
            'amp': peak amplitude,
            'snr': estimate signal to noise
                ratio,
            'y0_err': error in y0,
            'x0_err': error in x0,
            'I_err': error in I,
            'bg_err': error in bg,
            'converged': bool, whether
                the iteration converged
        }

    """
    # Subtract BG and divide out gain 
    psf_img = utils.rescale_img(psf_img, 
        camera_offset=camera_offset,
        camera_gain=camera_gain)

    # Make initial guess using radial symmetry
    guess = radial_symmetry(psf_img, sigma=sigma)

    # Current parameter estimate
    pars = np.array([guess['y0'], guess['x0'], 
        guess['I'], guess['bg']])

    # Update to the parameter estimate, used
    # to call convergence / divergence
    update = np.ones(4, dtype='float64')

    # Hessian of log-likelihood function
    H = np.zeros((4, 4), dtype='float64')

    # Continue iterating until max_iter reached
    # or convergence reached
    iter_idx = 0
    while iter_idx < max_iter:

        # Check for convergence
        if any(np.abs(update[:2]) < convergence_crit):
            break 

        # Calculate PSF model, Jacobian, and 
        # Hessian under Poisson noise model 
        model, J, H = utils.L_poisson(
            psf_img, pars, sigma=sigma)

        # Calculate gradient of log-likelihood
        # with respect to each model parameter
        grad = J.sum(axis=0)

        # Invert the Hessian
        H_inv = utils.invert_hessian(H, ridge=ridge)

        # Determine the update vector, the change
        # in parameters
        update = -damp*(H_inv @ grad)

        # Apply the update
        if debug:
            print(pars)
            utils.wireframe_overlay(psf_img, model)

        pars += update 
        iter_idx += 1

    # If estimate is diverging, fall back to original
    # guess
    converged = (np.abs(update[:2]) < divergence_crit).all()
    if not converged:
        pars = [guess['y0'], guess['x0'], \
            guess['I'], guess['bg']]

    # Estimate peak amplitude of Gaussian
    amp = utils.amp_from_I(pars[2], sigma=sigma)

    # Estimate SNR
    snr = utils.estimate_snr(psf_img, amp)

    if debug:
        print(pars)
        utils.wireframe_overlay(psf_img, model)

    return dict((('y0', pars[0]), ('x0', pars[1]), \
        ('I', pars[2]), ('bg', pars[3]), ('amp', amp),
        ('snr', snr), ('converged', converged)))

def ls_int_gaussian(psf_img, sigma=1.0):
    """
    Find the least-squares estimate for the parameters
    of an integrated 2D Gaussian PSF model, given
    a sample PSF image. This is equivalent to the
    maximum likelihood estimate in the presence of
    Gaussian noise.

    args
    ----
        psf_img : 2D ndarray

    returns
    -------
        dict : {
            'y0': estimated y center,
            'x0': estimated x center,
            'I': estimated PSF intensity,
            'bg': estimated background intensity,
            'amp': estimated peak PSF amplitude,
            'snr': estimated SNR
        }

    """
    pass

def ls_point_gaussian(psf_img, sigma=1.0):
    """
    Find the least-squares estimate for the parameters
    of a pointwise-evaluated 2D Gaussian PSF model,
    given a sample PSF image. This is equivalent to
    the maximum likelihood estimate for this model
    in the presence of Gaussian noise.

    While the pointwise Gaussian is less accurate
    than the integrated Gaussian model, it is somewhat
    faster to evaluate.

    args
    ----
        psf_img : 2D ndarray
        sigma : float, width of 2D Gaussian

    returns
    -------
        dict : {
            'y0': estimated y center,
            'x0': estimated x center,
            'I': estimated PSF intensity,
            'bg': estimated background intensity,
            'amp': estimated peak PSF amplitude,
            'snr': estimated SNR
        }

    """
    pass

def ls_log_gaussian(psf_img, sigma=1.0, camera_bg=0.0,
    camera_gain=1.0):
    """
    Find the least-squares estimate for the 
    parameters of a pointwise 2D Gaussian PSF
    model, assuming noise is log-normally distributed.

    While this noise model is fairly unrealistic,
    it admits a linear LS estimator for the 
    parabolic log intensities of a pointwise 
    Gaussian. As a result, it is very fast.

    However, it has the issue that it is biased-
    the log normality assumption tends to make
    it biased toward the center of the fitting 
    window. Use at your own risk.

    args
    ----
        psf_img : 2D ndarray
        sigma : float, the width of the Gaussian
                model

    returns
    -------
        dict : {
            'y0': estimated y center,
            'x0': estimated x center,
            'I': estimated PSF intensity,
            'bg': estimated background intensity,
            'amp': estimated peak PSF amplitude,
            'snr': estimated SNR
        }

    """
    # Subtract BG and divide out gain 
    psf_img = utils.rescale_img(psf_img, 
        camera_offset=camera_offset,
        camera_gain=camera_gain)

    # Common factors
    V = sigma**2
    V2 = V*2

    # Estimate background by taking the mean
    # of the outer ring of pixels
    bg = utils.ring_mean(psf_img)

    # Subtract background
    psf_img_sub = utils.set_neg_to_zero(psf_img-bg)

    # Avoid taking log of zero pixels
    nonzero = psf_img_sub > 0.0

    # Pixel indices
    Y, X = np.indices(psf_img.shape)
    Y = Y[nonzero]
    X = X[nonzero]

    # Log PSF above background
    log_psf = np.log(psf_img_sub[nonzero])

    # Response vector in LS problem
    R = log_psf + np.log(V2*np.pi) + (Y**2 + X**2)/V2

    # Independent matrix in LS problem
    M = np.asarray([np.ones(nonzero.sum()), V*Y, V*X]).T

    # Compute the LS parameter estimate
    ls_pars = utils.pinv(M) @ R 

    # Use the LS parameters to the find the 
    # spot center
    y0 = ls_pars[1]
    x0 = ls_pars[2]
    I = np.exp(ls_pars[0] + (y0**2+x0**2)/V2)

    # Estimate peak Gaussian amplitude
    amp = utils.amp_from_I(I, sigma=sigma)

    # Estimate SNR
    snr = utils.estimate_snr(psf_img, amp)

    return dict((('y0', y0), ('x0', x0), ('I', I),
        ('bg', bg), ('amp', amp), ('snr', snr)))

LOCALIZE_METHODS = {
    'centroid': centroid,
    'radial_symmetry': radial_symmetry,
    'mle_poisson': mle_poisson,
    'ls_int_gaussian': ls_int_gaussian,
    'ls_log_gaussian': ls_log_gaussian,
}

def localize_psf(psf_img, method=None, **kwargs):
    """
    Run a subpixel localization method on a single
    PSF subimage.

    args
    ----
        psf_img : 2D ndarray
        method : str, key to LOCALIZE_METHODS
        **kwargs : to method

    returns
    -------
        dict, parameter estimates for the subpixel
            localization.

            All methods return at minimum the following
            estimates:
                y0 : spot center in y
                x0 : spot center in x
                I : spot integrated intensity
                amp : spot amplitude
                snr : signal-to-noise ratio

    """
    # If no method passed, return an empty dict
    if method is None:
        return {}

    # Get the desired localization  method
    try:
        method_f = LOCALIZE_METHODS[method]
    except KeyError:
        raise RuntimeError('quot.localize.localize_psf: '\
            'method %s not found; options: %s' % (method, 
                ', '.join(LOCALIZE_METHODS.keys())))

    # Run the localization method
    return method_f(psf_img, **kwargs)

def localize_frame(img, detections, method=None, 
    w=9, **kwargs):
    """
    Run a subpixel localization method on all
    detections in an image.

    args
    ----
        img : 2D ndarray
        detections : pandas.DataFrame with
            columns `yd` and `xd`, the 
            coords of the detections
        method : str
        w : int, size of PSF subwindow 
        **kwargs : to method

    returns
    -------
        pandas.DataFrame, localizations.
            Includes `yd` and `xd`.

    """
    # If no method passed, return unmodified dataframe
    if method is None:
        return detections 

    # Remove detections too close to the edge 
    hw = w//2
    detections = utils.remove_edge_detections(
        detections, img.shape, hw)

    # Fit a PSF and return its center in terms
    # of the entire frame
    def fit_and_shift(img, location):
        """
        args
        ----
            img : 2D ndarray
            localization : (int, int)

        """
        psf_img = utils.get_slice(img, hw, location)
        fit_pars = localize_psf(psf_img, method=method,
            **kwargs)
        fit_pars['y0'] += location[0]-hw
        fit_pars['x0'] += location[1]-hw 
        fit_pars['yd'] = location[0]
        fit_pars['xd'] = location[1]
        return fit_pars 

    return pd.DataFrame(
        [fit_and_shift(img, tuple(coords)) for _, coords in \
            detections[['yd', 'xd']].iterrows()]
    )

def localize_file(path, config_path, t0=None, t1=None,
    verbose=False):
    """
    Detect and localize all spots in an image file 
    according to a set of configuration settings.

    args
    ----
        path : str, path to ND2 or TIF file
        config_path : str, path to YAML config
            file
        t0 : int, start frame
        t1 : int, stop frame
        verbose : bool, progress bar

    returns
    -------
        pandas.DataFrame

    """
    if t0 is None:
        t0 = 0

    # Read config settings
    config = qio.read_config(config_path)

    # Create an image file reader
    reader = qio.ImageFileReader(path)

    # Create an image filterer
    filterer = image_filter.SubregionFilterer(
        reader, None, start_iter=t0, stop_iter=t1,
        **config['filtering'])

    # Set up filtering, detection, and localization
    if 'localize_in_filtered' in config['localization'].keys() \
        and config['localization']['localize_in_filtered']:
        locs = (
            localize_frame(
                img,
                detect.detect(img, **config['detection']),
                **config['localization'],
            ) for img in filterer 
        )
    else:
        locs = (
            localize_frame(
                reader.get_frame(i+t0),
                detect.detect(img, **config['detection']),
                **config['localization'],
            ) for i, img in enumerate(filterer)
        )

    # Run filtering, detection, and localization
    locs = pd.concat(
        [l.assign(frame_idx=i) for i, l in enumerate(locs)],
        ignore_index=True, sort=False
    )

    # Adjust frame indices to account for start
    # frame
    locs['frame_idx'] += t0 

    return locs 

def detect_file(path, config_path, t0=None, t1=None,
    verbose=False):
    """
    Detect all spots in an image file according to
    a set of configuration settings.

    args
    ----
        path : str, path to ND2 or TIF file
        config_path : str, path to YAML config
            file
        t0 : int, start frame
        t1 : int, stop frame
        verbose : bool, progress bar

    returns
    -------
        pandas.DataFrame

    """
    if t0 is None:
        t0 = 0

    # Read the config settings
    config = qio.read_config(config_path)

    # Create an image file reader
    reader = qio.ImageFileReader(path)

    # Create an image filterer
    filterer = image_filter.SubregionFilterer(
        reader, None, start_iter=t0, stop_iter=t1,
        **config['filtering'])

    # Set up filtering + detection
    detections = (detect.detect(img, **config['detection']) \
        for img in filterer)

    # Show a progress bar, if desired
    if verbose:
        data = tqdm(enumerate(detections))
    else:
        data = enumerate(detections)

    # Run detection
    detections = pd.concat(
        [d.assign(frame_idx=i) for i, d in data],
        ignore_index=True, sort=False,
    )

    # Adjust frame index
    detections['frame_idx'] += t0 

    return detections 

