#!/usr/bin/env python
"""
__main__.py -- launch batch analyses from the command line

"""
import os
import click
from shutil import copyfile

from .read import read_config, save_config
from .core import track_directory, track_file

ACCEPTABLE_EXTS = [".nd2", ".tif", ".tiff"]

@click.command()
@click.argument("target_dir", type=str)
@click.argument("config_path", type=str)
@click.option("-e", "--ext", type=str, default=".nd2",
    help="default .nd2")
@click.option("-n", "--n_threads", type=int, default=1,
    help="default 1")
@click.option("-o", "--out_dir", type=str, default=None,
    help="default same as image files")
@click.option("-c", "--contains", type=str, default=None,
    help="default None")
def batch_track(target_dir, config_path, ext, n_threads,
    out_dir, contains):
    """
    Run batch detection and tracking on all image files in 
    a target directory.

    TARGET_DIR  :  a directory with image files

    CONFIG_PATH :  path to a TOML configuration file

    EXT         :  the extension to look for (eg .nd2, .tif)

    N_THREADS   :  the number of parallel threads to use

    OUT_DIR     :  directory for output CSV files

    CONTAINS    :  a substring that all target files must contain

    """
    # Load the experimental configuration
    try:
        config = read_config(config_path)
    except:
        raise RuntimeError("Could not load config file {}".format(config_path))

    # If passed an image file instead of a directory, just run tracking
    # on this file
    if os.path.isfile(target_dir) and (os.path.splitext(target_dir)[-1] in ACCEPTABLE_EXTS):

        # Specify output file
        if (not out_dir is None) and (".csv" in out_dir):
            out_csv = out_dir 
        elif (not out_dir is None):
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            out_csv = os.path.join(
                out_dir,
                "{}_trajs.csv".format(
                    os.path.splitext(os.path.basename(target_dir))[0]
                )
            )
        else:
            out_csv = "{}_trajs.csv".format(os.path.splitext(target_dir)[0])

        # Run tracking
        track_file(
            target_dir,
            out_csv=out_csv,
            progress_bar=True,
            **config
        )

    # Otherwise run tracking on matching image files in the passed
    # directory
    elif os.path.isdir(target_dir):
        track_directory(
            target_dir,
            ext=ext,
            num_workers=n_threads,
            out_dir=out_dir,
            contains=contains,
            **config
        )

    else:
        raise RuntimeError("No file or directory: {}".format(target_dir))

@click.command()
@click.argument("config_path", type=str)
def make_naive_config(config_path):
    """
    Save a naive config file with some stock settings to a target
    path. 

    """
    src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "stock_config.toml"
    )
    copyfile(src, config_path)

if __name__ == "__main__":
    batch_track()
