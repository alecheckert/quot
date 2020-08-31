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

# Parallelization
import dask 
from dask.diagnostics import ProgressBar 

# File readers and filterers
from .chunkFilter import ChunkFilter

# Core detection function
from .findSpots import detect 

# Core localization function
from .subpixel import localize_frame

# Core tracking function
from .track import track 

def localize_file(path, out_csv=None, progress_bar=True, **kwargs):
    """
    Run filtering, detection, and subpixel localization on 
    a single image movie. This does NOT perform tracking.

    args
    ----
        path        :   str, path to the image file
        out_csv     :   str, path to save file, if 
                        desired
        progress_bar:   bool, show a progress bar
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

        frames = enumerate(f)
        if progress_bar:
            frames = tqdm(frames)

        locs = []
        for frame_idx, frame in frames:

            # Find spots in this image frame
            detections = detect(frame, **kwargs['detect'])

            # Localize spots to subpixel resolution
            locs.append(localize_frame(frame, detections, 
                **kwargs['localize']).assign(frame=frame_idx))

        locs = pd.concat(locs, ignore_index=True, sort=False)

        # Adjust for start index
        locs['frame'] += kwargs['filter'].get('start', 0)

    # Save to a file, if desired
    if not out_csv is None:
        locs.to_csv(out_csv, index=False)

    return locs 

def track_file(path, out_csv=None, progress_bar=True, **kwargs):
    """
    Run filtering, detection, subpixel localization, and 
    tracking on a single target movie.

    args
    ----
        path        :   str, path to the image file
        out_csv     :   str, path to save file, if 
                        desired
        progress_bar:   bool, show a progress bar
        kwargs      :   tracking configuration

    returns
    -------
        pandas.DataFrame, the localizations indexed by 
            trajectory

    """ 
    # Run filtering + detection + localization
    locs = localize_file(path, out_csv=None, progress_bar=progress_bar,
        **kwargs)

    # Track localizations between frames
    locs = track(locs, **kwargs['track'])

    # Save to a file if desired 
    if not out_csv is None:
        locs.to_csv(out_csv, index=False)

    return locs 

def track_files(paths, num_workers=4, save=True, **kwargs):
    """
    Run tracking on several files using parallelization.

    args
    ----
        paths       :   list of str, paths to image files to track
        num_workers :   int, the number of threads to use
        save        :   bool, save the output to CSVs files. The names
                        for these CSVs are generated from the names of 
                        the corresponding image files.
        kwargs      :   tracking configuration, as read with 
                        quot.io.read_config

    returns
    -------
        list of pandas.DataFrame, the tracking results for each 
            file

    """
    # Tracking function for one file with lazy evaluation
    @dask.delayed 
    def driver(path):
        if save:
            out_csv = "{}_trajs.csv".format(os.path.splitext(path)[0])
        else:
            out_csv = None 
        return track_file(
            path,
            out_csv=out_csv,
            progress_bar=False,
            **kwargs
        )

    # Run localization and tracking on each file
    results = [driver(path) for path in paths]
    scheduler = "single-threaded" if num_workers == 1 else "processes"
    with ProgressBar():
        dask.compute(*results, scheduler=scheduler, num_workers=num_workers)

    return results 


def track_directory(path, ext='.nd2', num_workers=4, save=True, **kwargs):
    """
    Find all image files in a directory and run 
    localization and tracking.

    args
    ----
        path        :   str, path to directory
        ext         :   str, image file extension
        num_workers :   int, the number of threads
                        to use 
        save        :   bool, save the results to CSV
                        files
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

    # Run tracking and localization
    track_files(image_paths, num_workers=num_workers, save=save, **kwargs)

