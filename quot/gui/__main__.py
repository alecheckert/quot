#!/usr/bin/env python
"""
__main__.py

"""
# Core GUI utilities
import PySide2
from PySide2.QtWidgets import QApplication
import pyqtgraph

# Paths
import sys
import os 
from glob import glob 

# CLI
import click 

# GUIs
from .launcher import Launcher 
from .imageViewer import ImageViewer
from .detectViewer import DetectViewer 
from .spotViewer import SpotViewer 
from .attributeViewer import AttributeViewer 
from .trackViewer import TrackViewer 
from .masker import Masker 
from .maskInterpolator import MaskInterpolator

# Custom GUI utilities
from .guiUtils import set_dark_app, split_channels_nd2

def launch_gui(gui, *args, **kwargs):
    """
    Generate a QApplication instance and launch a GUI 
    with it.

    args
    ----
        gui     :   GUI to launch

    """
    pyqtgraph.Qt.QT_LIB = "PySide2"
    app = QApplication([])
    set_dark_app(app)
    instance = gui(*args, **kwargs)
    sys.exit(app.exec_())

@click.group()
def cli():
    pass

@cli.command()
def main():
    """
    Menu to select other, more specialized GUIs

    """
    launch_gui(Launcher)

@cli.command()
@click.argument('path', type=str)
@click.option('-f0', '--start_frame', type=int, default=0, help='default 0')
@click.option('-f1', '--stop_frame', type=int, default=100, help='default 100')
@click.option('-s', '--gui_size', type=int, default=900, help='default 900')
def detect(path, **kwargs):
    """
    Experiment with detection settings on one image file.

    """
    launch_gui(DetectViewer, path, **kwargs)

@cli.command()
@click.argument("image_path", type=str)
@click.argument("locs_path", type=str)
@click.option('-f0', '--start_frame', type=int, default=0, help='default 0')
@click.option('-s', '--gui_size', type=int, default=800, help='default 800')
def overlay(image_path, locs_path, **kwargs):
    """
    Simple overlay of localizations onto a movie.

    """
    launch_gui(SpotViewer, image_path, locs_path, **kwargs)

@cli.command()
@click.argument("locs_path", type=str)
@click.option("-n", "--max_spots", type=int, default=10000, help='default 10000')
@click.option("-s", "--gui_size", type=int, default=600, help="default 600")
def attributes(locs_path, **kwargs):
    """
    Make a scatter plot of localization attributes.

    """
    launch_gui(AttributeViewer, locs_path, **kwargs)

@cli.command()
@click.argument("image_path", type=str)
def image(image_path):
    """
    Simple image viewer for a TIF/ND2 movie

    """
    launch_gui(ImageViewer, image_path)

@cli.command()
@click.argument('image_path', type=str)
@click.argument('locs_path', type=str)
@click.option('-s', '--gui_size', type=int, default=600, help='default 600')
@click.option('-p', '--pixel_size_um', type=float, default=0.16, help='default 0.16')
@click.option('-f', '--frame_interval', type=float, default=0.00548, help='default 0.00548')
@click.option('-f0', '--start_frame', type=int, default=0, help='default 0')
@click.option('-f1', '--stop_frame', type=int, default=100, help='default 100')
def track(image_path, locs_path, **kwargs):
    """
    Change tracking settings in real time

    """
    launch_gui(TrackViewer, image_path, locs_path, **kwargs)

@cli.command()
@click.argument('image_path', type=str)
@click.option("-m", "--max_points_freestyle", type=int, default=20, help="default 20")
def mask(image_path, **kwargs):
    """
    Draw 2D masks on an movie or image

    """
    launch_gui(Masker, image_path, **kwargs)

@cli.command()
def mask_interpolator(**kwargs):
    """
    Draw 2D masks on a movie or image, interpolating masks between frames

    """
    launch_gui(MaskInterpolator, **kwargs)

@cli.command()
@click.argument("path", type=str)
@click.option("-o", "--out_dir", default=None, type=str, help="default input directory")
def split_channels(path, out_dir):
    """
    Split ND2 files into TIF files for each channel.

    """
    # If the output directory is passed and does not exist, make it
    if (not out_dir is None) and (not os.path.isdir(out_dir)):
        os.mkdir(out_dir)
        
    # If the input is a directory, get every ND2 file in this directory and
    # split them all
    if os.path.isdir(path):
        nd2_paths = glob(os.path.join(path, "*.nd2"))
        for nd2_path in nd2_paths:
            split_channels_nd2(nd2_path, out_dir=out_dir)

    # If the input is a file, split this file
    elif os.path.isfile(path) and os.path.splitext(path)[1] == ".nd2":
        split_channels_nd2(path, out_dir=out_dir)

    # Incompatible input
    else:
        raise RuntimeError("path {} is not a directory or ND2 file".format(path))

if __name__ == '__main__':
    cli()
