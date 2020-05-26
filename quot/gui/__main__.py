#!/usr/bin/env python
"""
__main__.py

"""
import sys 

# Paths
import os 

# CLI
import click 

# Core QApplication instance
from PySide2.QtWidgets import QApplication

# GUIs
from .launcher import Launcher 
from .imageViewer import ImageViewer
from .detectViewer import DetectViewer 
from .spotViewer import SpotViewer 
from .attributeViewer import AttributeViewer 
from .trackViewer import TrackViewer 
from .masker import Masker 

# Custom GUI utilities
from .guiUtils import set_dark_app

def launch_gui(gui, *args, **kwargs):
    """
    Generate a QApplication instance and launch a GUI 
    with it.

    args
    ----
        gui     :   GUI to launch

    """
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
def spot_viewer(image_path, locs_path, **kwargs):
    """
    Overlay localizations onto a movie.

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
def image_viewer(image_path):
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
def track_viewer(image_path, locs_path, **kwargs):
    """
    Change tracking settings in real time

    """
    launch_gui(TrackViewer, image_path, locs_path, **kwargs)

@cli.command()
@click.argument('image_path', type=str)
def mask(image_path, **kwargs):
    """
    Draw 2D masks on an movie or image

    """
    launch_gui(Masker, image_path, **kwargs)

if __name__ == '__main__':
    cli()
