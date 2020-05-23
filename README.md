# quot
A simple GUI to compare spot detection methods in single molecule microscopy data

## Installation

1. Clone the repository:
```
git clone https://github.com/alecheckert/quot.git
```

2. Create a `conda` environment for `quot`. (If you don't already have it, you'll need `conda`: https://docs.conda.io/en/latest/miniconda.html.) Navigate to the top-level `quot` directory and run 

```
    conda env create -f quot_env.yml
```

3. Switch to the `quot` environment with `conda activate quot_env`. 

4. Install the `quot` package. From the top-level `quot` directory, run

```
python setup.py develop
```

`quot` is still in active development. The `develop` option  will track changes in the source files as new versions become available.

## Example usage

`quot` does single molecule tracking with five steps:

    1. Read a frame from an image file
    2. (Optional) Filter the frame to remove background
    3. Find spots in the frame
    4. Localize spots to subpixel resolution
    5. Reconnect spots into trajectories

`sample_config.toml` is a sample `quot` configuration file that specifies parameters for each of these steps. To run tracking on a Nikon ND2 file with these settings, do

```
    from quot.read import read_config
    from quot.core import track_file

    # Specify target
    target_path = "spt_movie.nd2"

    # Read the configuration
    config = read_config("sample_config.toml")

    # Run localization and tracking on the file
    locs = track_file(target_path, **config)
```

## Running the `quot` GUI

The easiest way to start exploring SPT options in `quot` is to use the `quot` GUI. 

`quot` can be run on the command line on either ND2 or TIF files. To get usage information, use
```
quot --help
```

To launch a menu with options to option other GUIs, do
```
quot main
```

`quot` also has a variety of other commands. These are mostly shortcuts to lower-level GUIs. For example, to run the filtering/detection module on a specific file, do
```
quot detect samples/sample_movie.tif
```
