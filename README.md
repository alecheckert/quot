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

## Alternative install with `pip`

In the future `quot` will be installable with `pip` alone. This functionality is currently on the `quot_env_v2` branch. To install via this method, do:
```
  git clone https://github.com/alecheckert/quot.git
  git checkout quot_env_v2
  cd quot
  # Make a new venv or something here
  pip install -e .
```

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

An example SPT movie and config file are in the `quot/samples` directory.
To run the example, navigate to the `quot/samples` directory and run
```
    quot-track sample_movie.tif sample_config.toml -o sample_trajs.csv
```

The result should be saved to `samples/sample_trajs.csv`.

`quot-track` can also be used for batch tracking on directories with 
many image files. For example, suppose we have the following directory
structure:
```
  -> my_config.toml
  -> directory_with_nd2_files
     -> file_1.nd2
     -> file_2.nd2 
     -> file_3.nd2
```

To run batch tracking on all of the files in `directory_with_nd2_files`,
you could do:
```
    quot-track directory_with_nd2_files my_config.toml -o output_directory -n 3
```

The resulting `.csv` files will be placed in `output_directory` and will be
named based on their parent ND2 file. The `-n` (equivalently, `--n_threads`)
argument specifies how many threads to run in parallel.

As always, to get a full list of the options to `quot-track`, use
```
    quot-track --help
```

## Finding external hard drives

Some users have reported trouble finding external hard drives with the `quot` file selection dialogs. If this happens, try the following:

1. Look under `/Volumes` (if using macOS).
2. If the hard drive is not visible under `/Volumes`, navigate to the hard drive in the Terminal and launch an instance of the `quot` GUI on an image file there (for instance, `quot image some_random_file.nd2`). The file should be stored on the hard drive. On macOS Catalina, this triggers a permissions dialog that makes the hard drive subsequentialy visible in the `quot` file dialog selections.

If you continue to have trouble finding the external hard drives, contact <aheckert@berkeley.edu>.
