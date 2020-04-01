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

To launch the main GUI, do
```
quot main
```

To run the filtering/detection module on a specific file, do
```
quot gui samples/sample_movie.tif
```

Other options include running `quot` on a subregion of the input file, or only on specific frames. Once in the GUI, a settings file (for instance, `settings.yaml`) can be saved with the detection settings and then run on a file or directory of files with the command
```
quot loc samples settings.yaml
```

The GUI can also be launched from within Python using the `quot.gui.GUI` class:
```
from quot import gui

filename = 'samples/sample_movie.tif'
gui_inst = gui.GUI(filename)

# If we only want to run on rectangular subregion
gui_inst = gui.GUI(filename, subregion=[[100, 200], [150, 350]])

# If we only want to run on frames 120 through 220 
gui_inst = gui.GUI(filename, subregion=[[100, 200], [150, 350]], frame_limits=(120, 220))
```
