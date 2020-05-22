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
quot detect samples/sample_movie.tif
```

