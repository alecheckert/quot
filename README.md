# quot
A simple GUI to compare spot detection methods in single molecule microscopy data

## Installation

Clone the repository:
```
git clone https://github.com/alecheckert/quot.git
```

Navigate to the folder and run
```
python setup.py develop
```

`quot` is still in active development. The `develop` option  will track changes in the source files as new versions become available.

## Running `quot`

`quot` can be run on the command line on either ND2 or TIF files. To get usage information, use
```
quot --help
```

A typical usage is 
```
quot samples/sample_movie.tif
```

Other options include running `quot` on a subregion of the input file, or only on specific frames.

The GUI can also be launched from within Python using the `quot.gui.GUI` class:
```
from quot import gui

filename = 'samples/sample_movie.tif'
gui = gui.GUI(filename)

# If we only want to run on rectangular subregion
gui = gui.GUI(filename, subregion=[[100, 200], [150, 350]])

# If we only want to run on frames 120 through 220 
gui = gui.GUI(filename, subregion=[[100, 200], [150, 350]], frame_limits=(120, 220))
```
