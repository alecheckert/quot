"""
localize.py -- run filtering and detection on a full file

"""
# Numerics
import numpy as np 

# Dataframes
import pandas as pd 

# File reader 
from quot import qio 

# Image filtering utilities
from quot import image_filter 

# Detection utilities
from quot import detect 

# Progress bar
from tqdm import tqdm 

def loc_file(path, config_path, t0=None, t1=None,
    verbose=False):
    """
    Localize an image file according to a set of 
    configuration settings.

    args
    ----
        path : str, path to ND2 or TIF file
        config_path : str, path to YAML config
            file
        t0 : int, start frame
        t1 : int, stop frame
        verbose : bool, show progress bar

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

