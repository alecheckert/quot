#!/usr/bin/env python
"""
core.py -- high-level user functions for running 
filtering, detection, subpixel localization, and 
tracking sequentially on the same datasets

"""
# File paths
import os 
from glob import glob 

# Progress bar
from tqdm import tqdm 

# Dataframes
import pandas as pd 

# File readers and filterers
from .ChunkFilter import ChunkFilter

# Core detection function
from .spot import detect 

# Core localization function
from .subpixel import localize_frame

# Core tracking function
from .track import track 

def localize_file(path, out_csv=None, **kwargs):
    """
    Run filtering, detection, and subpixel localization on 
    a single image movie. This does NOT perform tracking.

    args
    ----
        path        :   str, path to the image file
        out_csv     :   str, path to save file, if 
                        desired
        kwargs      :   configuration

    returns
    -------
        pandas.DataFrame, the localizations

    """
    # Make sure the file exists
    assert os.path.isfile(path), "quot.__main__.localize_file: " \
        "file %s does not exist" % path 

    # Open an image file reader with some filtering
    # settings, if desired
    with ChunkFilter(path, **kwargs['filter']) as f:

        locs = []
        for frame_idx, frame in tqdm(enumerate(f)):

            # Find spots in this image frame
            detections = detect(frame, **kwargs['detect'])

            # Localize spots to subpixel resolution
            locs.append(localize_frame(frame, detections, 
                **kwargs['localize']).assign(frame=frame_idx))

        locs = pd.concat(locs, ignore_index=True, sort=False)

    # Save to a file, if desired
    if not out_csv is None:
        locs.to_csv(out_csv, index=False)

    return locs 

def track_file(path, out_csv=None, **kwargs):
    """
    Run filtering, detection, subpixel localization, and 
    tracking on a single target movie.

    args
    ----
        path        :   str, path to the image file
        out_csv     :   str, path to save file, if 
                        desired
        kwargs      :   configuration

    returns
    -------
        pandas.DataFrame, the localizations indexed by 
            trajectory

    """ 
    # Run filtering + detection + localization
    locs = loc_file(path, out_csv=None, **kwargs)

    # Track localizations between frames
    locs = track(locs, **kwargs['track'])

    # Save to a file if desired 
    if not out_csv is None:
        locs.to_csv(out_csv, index=False)

    return locs 

def track_directory(path, ext='.nd2', verbose=True, 
    **kwargs):
    """
    Find all image files in a directory and run 
    localization and tracking.

    args
    ----
        path        :   str, path to directory
        ext         :   str, image file extension
        verbose     :   bool
        kwargs      :   configuration

    returns
    -------
        None. Output of trackng is saved to files 
            with extension "_trajs.csv" in the same
            directory.

    """
    # Make sure the directory exists
    assert os.path.isdir(path), "quot.__main__.localize_directory: " \
        "directory %s does not exist" % path 

    # Find all image files in this directory
    image_paths = glob("%s/*%s" % (path, ext))

    # Format the save files
    out_csvs = [i.replace(ext, "_trajs.csv") for i in image_paths]

    # Run pipeline on each file
    for path, out_csv in zip(image_paths, out_csvs):
        trajs = track_file(path, out_csv=out_csv, **kwargs)
        if verbose: print("Finished with file %s..." % path)