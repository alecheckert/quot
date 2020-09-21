# quot
A simple GUI to compare spot detection and tracking methods in single molecule tracking data.

## Install

1. Clone the repository:
```
    git clone https://github.com/alecheckert/quot.git
```

2. Create a `conda` environment for `quot`. (If you don't already have it, you'll need `conda`: https://docs.conda.io/en/latest/miniconda.html.) Navigate to the top-level `quot` directory and run 

```
    conda env create -f quot_env.yml
```

3. Switch to the `quot_env` environment:

```
    conda activate quot_env
```

4. Finally, install the `quot` package. From the top-level `quot` directory, run

```
    python setup.py develop
```

`quot` is still in active development. The `develop` option  will track changes in the source files as new versions become available.

## Run the `quot` GUI

The easiest way to explore SPT options in `quot` is to use the GUI. To launch the GUI, first switch to the `quot` environment:

```
    conda activate quot_env
```

Then start the main GUI with

```
    quot main
```

To get additional usage information, use
```
    quot --help
```

Other `quot` commands are mostly shortcuts to lower-level GUIs. For example, to run the filtering/detection module on a specific file, do
```
    quot detect samples/sample_movie.tif
```

## Run localization and tracking with `quot`

`quot` performs single molecule tracking with five steps:

1. Read a frame from an image file
2. (Optional) Filter the frame to remove background
3. Find spots in the frame
4. Localize spots to subpixel resolution
5. Reconnect spots into trajectories

Exactly how each step is performed can be specified with a config file. `quot` uses Tom's Obvious, Minimal Language (TOML) format.

`sample_config.toml` is a `quot` configuration file. To use these settings to run localization and tracking on a Nikon ND2 file:

```
    from quot import read_config, track_file

    # Specify target (ND2 and TIF files supported)
    target_path = "samples/sample_movie.tif"

    # Read the configuration
    config = read_config("samples/sample_config.toml")

    # Run localization and tracking on the file
    locs = track_file(target_path, **config)
```

Batch tracking can also be run on directories with SPT movies using the `track_directory` command:

```
    from quot import track_directory

    # Run localization and tracking on each ND2 file in 
    # this directory, saving results as CSVs
    track_directory(
        "path/to/ND2/files",  # directory containing image files
        ext=".nd2",           # extension of target image files
        num_workers=4,        # number of threads to use
        save=True,            # save results to CSVs
        **config              # experimental configuration
    )

```

## Finding external hard drives

Some users have reported trouble finding external hard drives with the `quot` file selection dialogs. If this happens, try the following:

1. Look under `/Volumes` (if using macOS).
2. If the hard drive is not visible under `/Volumes`, navigate to the hard drive in the Terminal and launch an instance of the `quot` GUI on an image file there (for instance, `quot image some_random_file.nd2`). The file should be stored on the hard drive. On macOS Catalina, this triggers a permissions dialog that makes the hard drive subsequentialy visible in the `quot` file dialog selections.

If you continue to have trouble finding the external hard drives, contact <aheckert@berkeley.edu>.
