#!/usr/bin/env python 
"""
__main__.py

"""
# CLI
import click 

# GUI interface
from quot.gui import GUI, MainGUI

# File path stuff
import os 

# Find files with a given extension
from glob import glob 

# Image file reader
from quot.qio import ImageFileReader 
from quot import localize

@click.group()
def cli():
    pass 

@cli.command()
@click.argument('filename', type=str)
@click.option('-h', '--gui_height', default=500, type=int,
    help='default 500')
@click.option('-y0', default=0, type=int, help='default 0')
@click.option('-y1', default=None, type=int,
    help='default frame limit')
@click.option('-x0', default=0, type=int, help='default 0')
@click.option('-x1', default=None, type=int,
    help='default frame limit')
@click.option('-t0', default=0, type=int,
    help='default first frame')
@click.option('-t1', default=None, type=int,
    help='default last frame')
@click.option('-c', '--crosshair_len', type=int, 
    default=4, help='default 4 pixels')
def gui(
    filename, 
    gui_height,
    y0,
    y1,
    x0,
    x1,
    t0,
    t1,
    crosshair_len,
):
    """
    Optimize filtering and detection settings with a GUI

    """
    # Check the frame limits
    reader = ImageFileReader(filename)
    n_frames, N, M = reader.get_shape()
    reader.close()
    if y0 is None:
        y0 = 0 
    if y1 is None:
        y1 = N 
    if x0 is None:
        x0 = 0 
    if x1 is None:
        x1 = M 
    if t0 is None:
        t0 = 0 
    if t1 is None:
        t1 = n_frames 
    subregion = [[y0, y1], [x0, x1]]

    # Run the GUI
    gui_obj = GUI(filename, gui_height=gui_height,
        subregion=subregion, method='sub_median',
        frame_limits=(t0, t1), crosshair_len=crosshair_len)

@cli.command()
@click.argument('path', type=str)
@click.argument('config_file', type=str)
@click.option('-t0', type=int, default=None, help='default first frame')
@click.option('-t1', type=int, default=None, help='default last frame')
@click.option('-e', '--ext', type=str, default='.nd2',
    help='file extension; default *.nd2')
@click.option('-v', '--verbose/--no_verbose', default=True,
    help='default True')
def loc(
    path,
    config_file,
    t0,
    t1,
    ext,
    verbose,
):
    """
    Run filtering and detection on a file
    or directory with ND2 files.

    """
    # Find matching files
    if os.path.isdir(path):
        in_files = glob("%s/%s" % (path, ext))
    elif os.path.isfile(path):
        in_files = [path]
    else:
        raise RuntimeError("Could not find file/directory " \
            "%s" % path)

    # Run localization on each of the input files
    for f in in_files:

        # Format outfile name
        out_f = '%s_locs.csv' % os.path.splitext(f)[0]

        # Run localization
        locs = localize.localize_file(f, config_file, t0=t0, 
            t1=t1, verbose=verbose)

        # Save 
        locs.to_csv(out_f, index=False)
        if verbose: print('Finished %s...' % f)

@cli.command()
def main():
    """
    Launch the main GUI, with options for each 
    of the other steps.

    """
    main_gui = MainGUI()

if __name__ == '__main__':
    cli()
