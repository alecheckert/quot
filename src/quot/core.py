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

    # If the config file does not contain a "filter" section,
    # don't worry about it
    kwargs["filter"] = kwargs.get("filter", {})

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
        pandas.DataFrame, the reconnected localizations

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

def track_files(paths, num_workers=4, save=True, out_dir=None, **kwargs):
    """
    Run tracking on several files using parallelization.

    args
    ----
        paths       :   list of str, paths to image files to track
        num_workers :   int, the number of threads to use
        save        :   bool, save the output to CSVs files. The names
                        for these CSVs are generated from the names of 
                        the corresponding image files.
        out_dir     :   str, output directory
        kwargs      :   tracking configuration, as read with 
                        quot.io.read_config

    returns
    -------
        list of pandas.DataFrame, the tracking results for each 
            file

    """
    # Create the output directory if it does not already exist
    if (not out_dir is None) and (not os.path.isdir(out_dir)):
        os.mkdir(out_dir)

    # Tracking function for one file with lazy evaluation
    @dask.delayed 
    def driver(path):
        if save and (not out_dir is None):
            out_csv = os.path.join(
                out_dir,
                "{}_trajs.csv".format(os.path.splitext(os.path.basename(path))[0])
            )
        elif save and (out_dir is None):
            out_csv = "{}_trajs.csv".format(os.path.splitext(path)[0])
        else:
            out_csv = None 
        try:
            return track_file(
                path,
                out_csv=out_csv,
                progress_bar=False,
                **kwargs
            )
        except Exception as e:
            print("WARNING: Failed to analyze file {} due to exception:".format(path))
            print(e)
            return []

    # Run localization and tracking on each file
    results = [driver(path) for path in paths]
    scheduler = "single-threaded" if num_workers == 1 else "processes"
    with ProgressBar():
        dask.compute(*results, scheduler=scheduler, num_workers=num_workers)

    return results 

def track_directory(path, ext='.nd2', num_workers=4, save=True, contains=None,
    out_dir=None, **kwargs):
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
        contains    :   str, a substring that all image files 
                        are required to contain
        out_dir     :   str, directory for output CSV files
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

    # Only include image files with a substring, if specified
    if not contains is None:
        image_paths = [j for j in image_paths if contains in os.path.basename(j)]

    # Run tracking and localization
    track_files(image_paths, num_workers=num_workers, save=save, 
        out_dir=out_dir, **kwargs)

def retrack_file(path, out_csv=None, **kwargs):
    """
    Given an existing set of localizations or trajectories, (re)run tracking
    to reconstruct trajectories.

    args
    ----
        path        :   str, path to a *trajs.csv file
        out_csv     :   str, path to save the resulting trajectories, if 
                        desired
        kwargs      :   tracking configuration

    returns
    -------
        pandas.DataFrame, the reconnected localizations 

    """
    # Load the file
    T = pd.read_csv(path)

    # Track localizations between frames
    T = track(T, **kwargs)

    # Save to a file, if desired
    if not out_csv is None:
        T.to_csv(out_csv, index=False)

    return T 

def retrack_files(paths, out_suffix=None, num_workers=1, **kwargs):
    """
    Given a set of localizations, run retracking on each file and save to a 
    CSV.

    If *out_suffix* is not specified, then the trajectories are saved to the 
    original set of localization files (overwriting them).

    args
    ----
        paths       :   list of str, a set of CSV files encoding trajectories
        out_suffix  :   str, the suffix to use when generating the output 
                        paths. If *None*, then the output trajectories are 
                        saved to the original file path.
        num_workers :   int, the number of threads to use
        kwargs      :   tracking configuration

    """
    # Avoid redundant extensions
    if (not out_suffix is None) and (not ".csv" in out_suffix):
        out_suffix = "{}.csv".format(out_suffix)

    @dask.delayed 
    def task(fn):
        """
        Retrack one file.

        """
        out_csv = fn if out_suffix is None else \
            "{}_{}".format(os.path.splitext(fn)[0], out_suffix)
        retrack_file(fn, out_csv=out_csv, **kwargs["track"])

    # Run retracking on all files
    scheduler = "single-threaded" if num_workers == 1 else "processes"
    tasks = [task(fn) for fn in paths]
    with ProgressBar():
        dask.compute(*tasks, num_workers=num_workers, scheduler=scheduler)
