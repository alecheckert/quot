#!/usr/bin/env python 
"""
__main__.py

"""
# CLI
import click 

# GUI interface
from quot.gui import GUI 

# Image file reader
from quot.qio import ImageFileReader 

@click.command()
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
def run_gui(
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

if __name__ == '__main__':
    run_gui()
