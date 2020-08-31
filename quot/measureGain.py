#!/usr/bin/env python
"""
measureGain.py -- measure camera gain and BG with a simple
linear gain model

"""
# Numeric
import numpy as np 

# Fit gain model with LS
from scipy.optimize import curve_fit 

# Image file reader
from .read import ImageReader 

# Plotting utilities
from .plot import plot_pixel_mean_variance

def measure_camera_gain(*nd2_files, start_frame=100, plot=True):
    """
    Measure the camera gain and offset from a set of background movies.
    These can be used as the "camera_gain" and "camera_bg" arguments
    for localization settings, allowing the user to retrieve the PSF
    intensities in terms of photons rather than grayvalues.

    Each file in *nd2_files* should be a movie of an unlabeled, defocused,
    empty coverslip. Nothing should be movie, blinking, or exhibiting anything
    other than the ordinary camera noise. The movie should be acquired
    with exactly the same camera/stroboscopic settings as the SPT movies.

    Ideally, each movie in *nd2_files* should use a different level of 
    illumination, which facilitates more accurate measurement of the 
    gain.

    The method uses a pure Poisson noise model that is most applicable
    to EMCCD cameras.

    args
    ----
        nd2_files       :   variable number of str, background movies
        start_frame     :   int, the first frame in each movie to 
                            consider. This prevents photobleaching in
                            the early frames from influencing the 
                            result.

    returns
    -------
        dict, floats keyed to "camera_gain" and "camera_bg"

    """
    means, variances = [], []
    for i, nd2_file in enumerate(nd2_files):

        # Determine whether there are enough frames in this movie
        reader = ImageReader(nd2_file)
        if reader.n_frames <= start_frame:
            print("Warning: file {} only has {} frames".format(nd2_file, reader.n_frames))
            reader.close()
            continue 

        # Accumulate pixel means and variances for this movie
        I = np.zeros((reader.height, reader.width), dtype=np.float64)
        I2 = np.zeros((reader.height, reader.width), dtype=np.float64)
        n_frames = reader.n_frames - start_frame 
        for frame_idx in range(start_frame, reader.n_frames):
            frame = reader.get_frame(frame_idx)
            I += frame/n_frames 
            I2 += (frame**2)/n_frames 

        reader.close()

        # Pixel variance 
        I2 -= (I**2)

        # Append to the global list of means / variances
        means.append(I.ravel().copy())
        variances.append(I2.ravel().copy())

    # Flatten
    means_flat = np.concatenate(means)
    variances_flat = np.concatenate(variances)

    def linear_model(pixel_mean, gain, bg):
        """
        Given pixel mean, return pixel variance given a particular
        *gain* and *bg*.

        """
        return gain * (pixel_mean - bg)

    # Fit observed means and variances to a linear gain model
    popt, pcov = curve_fit(linear_model, means_flat, variances_flat, 
        bounds=(np.array([0, 0]), np.array([np.inf, np.inf])),
        p0=np.array([100.0, 400.0]))

    # Show the result to the user
    if plot:
        plot_pixel_mean_variance(means, variances, origin_files=nd2_files,
            model_gain=popt[0], model_bg=popt[1])

    return dict((
        ("camera_gain", popt[0]),
        ("camera_bg", popt[1])
    ))
