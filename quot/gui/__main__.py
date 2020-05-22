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
from .Launcher import Launcher 
from .DetectViewer import DetectViewer 

# Custom GUI utilities
from .guiUtils import set_dark_app

@click.group()
def cli():
    pass

@cli.command()
def main():
    """
    Menu to select other, more specialized GUIs

    """
    app = QApplication()
    set_dark_app(app)
    gui = Launcher()
    sys.exit(app.exec_())

@cli.command()
@click.argument('path', type=str)
@click.option('-f0', '--start_frame', type=int, default=0, help='default 0')
@click.option('-f1', '--stop_frame', type=int, default=100, help='default 100')
@click.option('-s', '--gui_size', type=int, default=900, help='default 900')
def detect(path, **kwargs):
    """
    Experiment with detection settings on one image file.

    """
    app = QApplication()
    set_dark_app(app)
    gui = DetectViewer(path, **kwargs)
    sys.exit(app.exec_())

if __name__ == '__main__':
    cli()
