"""
gui.py

"""
# tkinter GUI
import tkinter 
from tkinter import filedialog 

# File paths 
import os 
from glob import glob 

# Numeric
import numpy as np 

# Dataframes
import pandas as pd 

# Plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from matplotlib.path import Path 

# Pillow-style images, expected by tkinter
import PIL.Image, PIL.ImageTk 

# Filtering stuff
from scipy import ndimage as ndi 

# Hard copy 
from copy import copy 

# File reader 
from quot import qio 

# Image filtering/BG subtraction utilities
from quot.image_filter import SubregionFilterer

# Detection functions
from quot import detect 

# Localization functions
from quot import localize 

# Tracking functions
from quot import track 

# Visualize functions
from quot import visualize 

# Utilities
from quot import utils 
from .utils import set_neg_to_zero, overlay_spots, \
    label_binary_spots, upsample_overlay_two_color, \
    upsample

class GUI(object):
    """

    init
    ----
        filename : str, path to a TIF or ND2 file readable
            by quot.qio.ImageFileReader

        subregion : [(int, int), (int, int)], the limits
            of a rectangular subregion in the form 
            [(y0, y1), (x0, x1)]

        method : str, the default filtering method. Can 
            change with sliders later.

        gui_height : int, the height of the GUI. This will
            vary depending on the system. The class automatically
            tries to resize the image to fit in this frame, but
            this is somewhat of an art...

        frame_limits : (int, int), the lower and upper
            frames to allow in the frame slider (can 
            be useful to look at frames closer together)

        crosshair_len : int, the length of the crosshairs

    """
    def __init__(
        self,
        filename,
        subregion=None,
        method='sub_median',
        gui_height=500,
        frame_limits=None,
        crosshair_len=4,
        root=None,
    ):
        self.filename = filename
        self.gui_height = gui_height
        self.method = method 
        self.crosshair_len = crosshair_len

        # Image file reader
        self.reader = qio.ImageFileReader(self.filename)
        self.n_frames, self.N, self.M = self.reader.get_shape()

        # If no subregion is passed, default to the full frame
        if subregion is None:
            self.subregion = [[0, self.N], [0, self.M]]
        else:
            self.subregion = subregion 

        # If no frame limits are passed, default to the 
        # first and last frame
        if frame_limits is None:
            self.frame_limits = (0, self.n_frames)
        else:
            self.frame_limits = frame_limits

        # Set the positions and sizes of each of the images

        # Height and width of an individual image in pixels
        IH = self.subregion[0][1] - self.subregion[0][0]
        IW = self.subregion[1][1] - self.subregion[1][0]

        # Subregion aspect ratio
        AR = IH / IW 

        # Set GUI width
        self.gui_width = int(self.gui_height / AR)+1
        self.gui_size = (self.gui_height, self.gui_width)

        # y: the dividing line between the two columns of images
        y = self.gui_width // 2
        x = self.gui_height // 2

        # The positions of each image in the frame (frame_0)
        self.image_pos = [
            [0, 0], [0, x], [y, 0], [y, x]
        ]

        # The image resizing factor, chosen so that the images
        # fill the frame
        self.image_resize = (0.5 * self.gui_height / IH)

        # Image filtering kwargs; default to none
        self.filter_kwargs = {}

        # Image filterer
        self.filterer = SubregionFilterer(self.reader, 
            subregion=self.subregion, method=self.method)


        ## TKINTER INTERFACE COMPARTMENTALIZATION

        # Instantiate the main tkinter interface
        if root is None:
            self.root = tkinter.Tk()
        else:
            self.root = root 
        self.root.title("Optimize detection")

        # Master frame, containing both the upper
        # frame (self.frame_0) for images and the
        # lower frame (self.frame_1) for options
        self.master_frame = tkinter.Frame(self.root,
            height=self.gui_size[0], width=self.gui_size[1])
        self.master_frame.pack()

        ## FRAME 0: for images
        self.frame_0 = tkinter.Frame(self.master_frame)
        self.frame_0.pack()
        self.canvas = tkinter.Canvas(self.frame_0, 
            height=self.gui_size[0], width=self.gui_size[1])
        self.canvas.pack()

        ## FRAME 1: for options
        self.frame_1 = tkinter.Frame(self.master_frame)
        self.frame_1.pack(pady=10, padx=10)


        ## OPTION VARIABLES

        # The type of filtering approach, which responds 
        # to menu self.M0
        self.filter_type = tkinter.StringVar(self.root)
        self.filter_type.set(self.method)
        self.filter_type.trace('w', self._change_filter_type_callback)

        # The type of detection approach, which responds
        # to menu self.M1
        self.detect_type = tkinter.StringVar(self.root)
        self.detect_type.set('simple_gauss')
        self.detect_type.trace('w', self._change_detect_type_callback)

        # The block size, the temporal size of the block
        # used for image filtering
        self.block_size_var = tkinter.IntVar(self.root)
        self.block_size_var.set(60)
        self.block_size_var.trace('w', self._change_block_size_callback)

        # Default detection function and corresponding kwargs
        self.detect_f = DETECT_METHODS[self.detect_type.get()]
        self.detect_kwargs = DETECT_KWARGS_DEFAULT[self.detect_type.get()]

        # The current frame index, which responds to 
        # slider self.s00
        self.frame_idx = self.frame_limits[0]

        # The current image vmax, which responds to 
        # slider self.s10
        self.vmax = 255.0

        # A semantically flexible variable that responds 
        # to slider self.s20, for changes in the filter type
        self.k0 = 1

        # A semantically flexible variable that responds
        # to slider self.s30, for changes in the filter type
        self.k1 = 1

        # A semantically flexible variable that responds 
        # to slider self.s01, for changes in the detect type
        self.k2 = 1

        # A semantically flexible variable that responds
        # to slider self.s11, for changes in the detect type
        self.k3 = 1

        # A semantically flexible variable that responds 
        # to slider self.s21, for changes in the detect type
        self.k4 = 1

        # A semantically flexible variable that responds 
        # to slider self.s31, for changes in the detect type
        self.k5 = 1


        ## OPTION MENUS

        # Menu that chooses the filter choice, informing
        # self.filter_type
        filter_choices = FILTER_KWARG_MAP.keys()
        self.M0 = tkinter.OptionMenu(self.frame_1, self.filter_type,
            *filter_choices)
        self.M0.grid(row=0, column=0, pady=5, padx=5)

        # Menu that chooses the detection method choice,
        # informing self.detect_type
        detect_choices = DETECT_METHODS.keys()
        self.M1 = tkinter.OptionMenu(self.frame_1, self.detect_type,
            *detect_choices)
        self.M1.grid(row=0, column=1, pady=5, padx=5)

        # Menu that chooses the block size, informing 
        # self.block_size_var
        block_size_choices = [5, 10, 20, 40, 60, 100, 200, 300]
        m2_label = tkinter.Label(self.frame_1, text='Filter block size')
        m2_label.grid(row=0, column=2, sticky=tkinter.W, pady=5, padx=5)
        self.M2 = tkinter.OptionMenu(self.frame_1, self.block_size_var,
            *block_size_choices)
        self.M2.grid(row=1, column=2, pady=5, padx=5)

        # Button to show detections, if desired
        button_kwargs = {
            'activeforeground': '#dadde5',
            'activebackground': '#455462',
        }
        self.B0 = tkinter.Button(self.frame_1, text='Overlay detections',
            command=self._overlay_detections_callback, **button_kwargs)
        self.B0.grid(row=2, column=2, pady=5, padx=5)

        # The current state of this button
        self.B0_state = False 

        # Button to resize the threshold bar
        self.B1 = tkinter.Button(self.frame_1, text='Rescale threshold',
            command=self._rescale_threshold_callback, **button_kwargs)
        self.B1.grid(row=3, column=2, pady=5, padx=5)

        # Button to save the current configurations
        self.B2 = tkinter.Button(self.frame_1, text='Save current settings',
            command=self._save_settings_callback, **button_kwargs)
        self.B2.grid(row=4, column=2, pady=5, padx=5)


        ## SLIDER LABELS, which change in response to 
        ## different filtering / detection methods 
        label_kwargs = {'sticky': tkinter.W, 'pady': 5,
            'padx': 5}

        # slider label SL00, for slider S00 (the 
        # frame index). Value of this label is 
        # stored in SL00_v.
        self.SL00_v = tkinter.StringVar(self.root)
        self.SL00 = tkinter.Label(
            self.frame_1, 
            textvariable=self.SL00_v
        )
        self.SL00.grid(row=1, column=0, **label_kwargs)
        self.SL00_v.set('Frame index')

        # slider label SL10, for slider S10 (vmax).
        # Value of this label is stored in SL10_v.
        self.SL10_v = tkinter.StringVar(self.root)
        self.SL10 = tkinter.Label(
            self.frame_1, 
            textvariable=self.SL10_v
        )
        self.SL10.grid(row=3, column=0, **label_kwargs)
        self.SL10_v.set('vmax')

        # slider label SL20, for slider S20 (responds
        # to the filter type). Value of this label is
        # stored in SL20_v.
        self.SL20_v = tkinter.StringVar(self.root)
        self.SL20 = tkinter.Label(
            self.frame_1, 
            textvariable=self.SL20_v
        )
        self.SL20.grid(row=5, column=0, **label_kwargs)

        # slider label SL30, for slider S30 (responds
        # to the filter type). Value of this label is
        # stored in SL30_v.
        self.SL30_v = tkinter.StringVar(self.root)
        self.SL30 = tkinter.Label(
            self.frame_1, 
            textvariable=self.SL30_v
        )
        self.SL30.grid(row=7, column=0, **label_kwargs)       

        # slider label SL01, for slider S01 (responds
        # to detection type). Value of this label is
        # stored in SL01_v.
        self.SL01_v = tkinter.StringVar(self.root)
        self.SL01 = tkinter.Label(
            self.frame_1, 
            textvariable=self.SL01_v
        )
        self.SL01.grid(row=1, column=1, **label_kwargs)              

        # slider label SL11, for slider S11 (responds
        # to detection type). Value of this label is
        # stored in SL11_v.
        self.SL11_v = tkinter.StringVar(self.root)
        self.SL11 = tkinter.Label(
            self.frame_1, 
            textvariable=self.SL11_v
        )
        self.SL11.grid(row=3, column=1, **label_kwargs)                     

        # slider label SL21, for slider S21 (responds
        # to detection type). Value of this label is
        # stored in SL21_v.
        self.SL21_v = tkinter.StringVar(self.root)
        self.SL21 = tkinter.Label(
            self.frame_1, 
            textvariable=self.SL21_v
        )
        self.SL21.grid(row=5, column=1, **label_kwargs)                     

        # slider label SL31, for slider S31 (responds
        # to detection type). Value of this label is 
        # stored in SL31_v.
        self.SL31_v = tkinter.StringVar(self.root)
        self.SL31 = tkinter.Label(
            self.frame_1, 
            textvariable=self.SL31_v
        )
        self.SL31.grid(row=7, column=1, **label_kwargs)

        # Iterable lists of slider labels
        self.filter_slider_labels = [self.SL20_v, self.SL30_v]
        self.detect_slider_labels = [self.SL01_v, self.SL11_v,
            self.SL21_v, self.SL31_v]


        ## SLIDERS
        slider_kwargs = {'orient': tkinter.HORIZONTAL,
            'activebackground': '#000000', 'resolution': 1,
            'length': 200, 'font': ('Helvetica', '16'),
            'sliderrelief': tkinter.FLAT, 'bd': 0}
        s_grid_kwargs = {'sticky': tkinter.W, 'pady': 5, 
            'padx': 5}

        # slider S00, which informs self.frame_idx
        self.S00 = tkinter.Scale(self.frame_1, from_=self.frame_limits[0], 
            to=self.frame_limits[1]-1, command=self._set_frame_idx,
            **slider_kwargs)
        self.S00.grid(row=2, column=0, **s_grid_kwargs)
        self.S00.set(0)

        # slider S10, which informs self.vmax
        self.S10 = tkinter.Scale(self.frame_1, from_=10.0,
            to=255.0, command=self._set_vmax,
            **slider_kwargs)
        self.S10.grid(row=4, column=0, **s_grid_kwargs)
        self.S10.set(255.0)

        # slider S20, which informs self.k0. Responds to 
        # changes in the filter type
        v = self.filter_type.get()
        self.S20 = tkinter.Scale(self.frame_1, from_=0.0, 
            to=5.0, command=self._set_k0, **slider_kwargs)
        self.S20.grid(row=6, column=0, **s_grid_kwargs)
        if len(FILTER_KWARG_MAP[v]) > 0:
            self.S20.set(FILTER_KWARGS_DEFAULT[v][FILTER_KWARG_MAP[v][0]])
        else:
            self.S20.set(1.0)

        # slider S30, which informs self.k1. Responds to 
        # changes in the filter type
        self.S30 = tkinter.Scale(self.frame_1, from_=0.0, 
            to=5.0, command=self._set_k1, **slider_kwargs)
        self.S30.grid(row=8, column=0, **s_grid_kwargs)
        if len(FILTER_KWARG_MAP[v]) > 1:
            self.S30.set(FILTER_KWARGS_DEFAULT[v][FILTER_KWARG_MAP[v][1]])
        else:
            self.S30.set(1.0)

        # slider S01, which informs self.k2. Responds to 
        # changes in the detect type
        v = self.detect_type.get()
        self.S01 = tkinter.Scale(self.frame_1, from_=0.0, 
            to=5.0, command=self._set_k2, **slider_kwargs)
        self.S01.grid(row=2, column=1, **s_grid_kwargs)
        if len(DETECT_KWARG_MAP[v]) > 0:
            self.S01.set(DETECT_KWARGS_DEFAULT[v][DETECT_KWARG_MAP[v][0]])
        else:
            self.S01.set(1.0)

        # slider S11, which informs self.k3. Responds to 
        # changes in the detect type
        self.S11 = tkinter.Scale(self.frame_1, from_=0.0, 
            to=5.0, command=self._set_k3, **slider_kwargs)
        self.S11.grid(row=4, column=1, **s_grid_kwargs)
        if len(DETECT_KWARG_MAP[v]) > 1:
            self.S11.set(DETECT_KWARGS_DEFAULT[v][DETECT_KWARG_MAP[v][1]])
        else:
            self.S11.set(1.0)

        # slider S21, which informs self.k4. Responds to 
        # changes in the detect type
        self.S21 = tkinter.Scale(self.frame_1, from_=0.0, 
            to=5.0, command=self._set_k4, **slider_kwargs)
        self.S21.grid(row=6, column=1, **s_grid_kwargs)
        if len(DETECT_KWARG_MAP[v]) > 2:
            self.S21.set(DETECT_KWARGS_DEFAULT[v][DETECT_KWARG_MAP[v][2]])
        else:
            self.S21.set(1.0)

        # slider S31, which informs self.k5. Responds to 
        # changes in the detect type
        self.S31 = tkinter.Scale(self.frame_1, from_=0.0, 
            to=5.0, command=self._set_k5, **slider_kwargs)
        self.S31.grid(row=8, column=1, **s_grid_kwargs)
        if len(DETECT_KWARG_MAP[v]) > 3:
            self.S31.set(DETECT_KWARGS_DEFAULT[v][DETECT_KWARG_MAP[v][3]])
        else:
            self.S31.set(1.0)


        ## GRID INITIALIZATION

        # Update sliders to be in accord with the current
        # filter type
        self._update_filter_sliders()

        # Update sliders to be in accord with the current
        # detect type
        self._update_detect_sliders()

        # Regenerate the primary photo
        self.img_00 = self.reader.frame_subregion(
            self.frame_idx,
            y0=self.subregion[0][0], y1=self.subregion[0][1],
            x0=self.subregion[1][0], x1=self.subregion[1][1]
        ).astype('float64')
        self._filter()
        self._detect()
        self._regenerate_photos()
        self._regenerate_primary_photo()

        # Run main loop
        self.root.mainloop()

    def _change_filter_type_callback(self, *args):
        self._update_filter_sliders()

    def _change_detect_type_callback(self, *args):
        self._update_detect_sliders()

    def _change_block_size_callback(self, *args):
        self._change_block_size()

    def _overlay_detections_callback(self, *args):
        self.B0_state = not self.B0_state 
        self._regenerate_primary_photo()
        self._regenerate_photos()

    def _rescale_threshold_callback(self, *args):
        self._rescale_threshold()

    def _save_settings_callback(self, *args):
        self._save_settings()

    def _update_filter_sliders(self):
        """
        Use the current value of self.filter_type to 
        update slider self.s20 and self.s30.

        """
        # Get the name of the current filtering method
        v = self.filter_type.get()

        # Set up the image filterer for this method
        self.filter_kwargs = FILTER_KWARGS_DEFAULT[v]
        self.filterer._change_filter_method(v, **self.filter_kwargs)

        # Set the slider names
        try:
            self.SL20_v.set(FILTER_KWARG_MAP[v][0])
            self.SL30_v.set(FILTER_KWARG_MAP[v][1])
        except IndexError:
            pass
        m = len(FILTER_KWARG_MAP[v])
        for j in range(m):
            self.filter_slider_labels[j].set(FILTER_KWARG_MAP[v][j])
        for j in range(m, 2):
            self.filter_slider_labels[j].set('')

        # Set the slider ranges 
        try:
            self.S20.configure(
                from_=FILTER_SLIDER_LIMITS[v][0][0],
                to=FILTER_SLIDER_LIMITS[v][0][1],
                resolution=FILTER_SLIDER_RESOLUTIONS[v][0]
            )
            self.S20.set(FILTER_KWARGS_DEFAULT[v][FILTER_KWARG_MAP[v][0]])
            self.S30.configure(
                from_=FILTER_SLIDER_LIMITS[v][1][0],
                to=FILTER_SLIDER_LIMITS[v][1][1],
                resolution=FILTER_SLIDER_RESOLUTIONS[v][1]
            )
            self.S30.set(FILTER_KWARGS_DEFAULT[v][FILTER_KWARG_MAP[v][1]])
        except IndexError:
            pass 

        # Run filtering, detection, and update canvas plots
        self._filter()
        self._detect()
        self._regenerate_photos()

    def _update_detect_sliders(self):
        """
        Update sliders S01, S11, S21, and S31 to 
        reflect the current detection method, 
        stored at self.detect_type.

        """
        # Get the name of the current detection method
        v = self.detect_type.get()

        # Set up the detection method
        self.detect_f = DETECT_METHODS[v]
        self.detect_kwargs = DETECT_KWARGS_DEFAULT[v]

        # Set the slider names 
        m = len(DETECT_KWARG_MAP[v])
        for j in range(m):
            self.detect_slider_labels[j].set(DETECT_KWARG_MAP[v][j])
        for j in range(m, 4):
            self.detect_slider_labels[j].set('')

        # Set the slider ranges
        try:
            self.S01.configure(
                from_=DETECT_SLIDER_LIMITS[v][0][0],
                to=DETECT_SLIDER_LIMITS[v][0][1],
                resolution=DETECT_SLIDER_RESOLUTIONS[v][0]
            )
            self.S01.set(DETECT_KWARGS_DEFAULT[v][DETECT_KWARG_MAP[v][0]])
            self.S11.configure(
                from_=DETECT_SLIDER_LIMITS[v][1][0],
                to=DETECT_SLIDER_LIMITS[v][1][1],
                resolution=DETECT_SLIDER_RESOLUTIONS[v][1]
            )
            self.S11.set(DETECT_KWARGS_DEFAULT[v][DETECT_KWARG_MAP[v][1]])
            self.S21.configure(
                from_=DETECT_SLIDER_LIMITS[v][2][0],
                to=DETECT_SLIDER_LIMITS[v][2][1],
                resolution=DETECT_SLIDER_RESOLUTIONS[v][2]
            )
            self.S21.set(DETECT_KWARGS_DEFAULT[v][DETECT_KWARG_MAP[v][2]])
            self.S31.configure(
                from_=DETECT_SLIDER_LIMITS[v][3][0],
                to=DETECT_SLIDER_LIMITS[v][3][1],
                resolution=DETECT_SLIDER_RESOLUTIONS[v][3]
            )
            self.S31.set(DETECT_KWARGS_DEFAULT[v][DETECT_KWARG_MAP[v][3]])
        except IndexError:
            pass

        # Run detection and update the canvas 
        self._detect()
        self._regenerate_photos()

    def _change_block_size(self):
        """
        Update the filterer's block size argument
        in response to change in self.block_size_var,
        set by the option menu self.M2.

        """
        # Get the current value of the block size 
        block_size = int(self.block_size_var.get())

        # Set the block size of the underlying filterer
        self.filterer._set_block_size(block_size)

        # Update the frame
        self._filter()
        self._detect()
        self._regenerate_photos()

    def _rescale_threshold(self):
        """
        Rescale the threshold slider to match the min/max 
        of the current filtered image.

        The identity of the threshold slider (i.e. which of
        self.S01 through self.S31 is assigned the threshold
        kwarg) will depend on the detection method.

        """
        # The current detection method
        v = self.detect_type.get()

        # Figure out if the current detection method 
        # admits a threshold ('t') kwarg. If not, do nothing.
        try:
            # The slider index corresponding to the threshold
            s_idx = DETECT_KWARG_MAP[v].index('t')

            # Set the limits of the corresponding slider to 
            # the minimum and maximum values of the current
            # filtered image (self.img_10)
            [self.S01, self.S11, self.S21, self.S31][s_idx].configure(
                from_=int(np.floor(self.img_10.min())),
                to=int(np.ceil(self.img_10.max()))
            )
        except ValueError:
            pass

    def _save_settings(self):
        """
        Prompt the user to enter a filename to save 
        the current detection settings.

        """
        # Choose the default directory prompt
        if hasattr(self, 'save_file'):
            initialdir = os.path.dirname(self.save_file)
        else:
            initialdir = os.getcwd()

        # Launch the file dialog GUI
        self.save_file = filedialog.asksaveasfilename(
            parent=self.root,
            initialdir=initialdir,
            defaultextension='.yaml',
        )

        # Format config settings for filtering and detection
        settings = {
            'filtering': {
                'method': self.filter_type.get(),
                'block_size': self.block_size_var.get(),
                **self.filter_kwargs,
            },
            'detection': {
                'method': self.detect_type.get(),
                **self.detect_kwargs,
            },
        }

        # Save
        qio.save_config(self.save_file, settings)

    def _filter(self):
        """
        Refilter the current frame according 
        to the current filtering settings.

        """
        self.img_01 = self.filterer.filter_frame(
            self.frame_idx,
            **self.filter_kwargs,
        )

    def _detect(self):
        """
        Run detection on the current filtered image
        according to the current detection settings.

        """
        self.img_10, self.img_11, positions = \
            self.detect_f(self.img_01, **self.detect_kwargs)

    def _set_filter_choice(self, *args):
        """
        Change the GUI in response to a change in the 
        self.filter_type variable.

        """
        self._update_filter_sliders()

    def _set_frame_idx(self, frame_idx):
        self.frame_idx = int(frame_idx)
        self.img_00 = self.reader.frame_subregion(
            self.frame_idx, 
            y0=self.subregion[0][0], y1=self.subregion[0][1],
            x0=self.subregion[1][0], x1=self.subregion[1][1]
        ).astype('float64')
        self._filter()
        self._detect()
        self._regenerate_photos()
        self._regenerate_primary_photo()

    def _set_vmax(self, vmax):
        self.vmax = float(vmax)
        self._regenerate_photos()

    def _set_k0(self, value):
        self.k0 = float(value)
        v = self.filter_type.get()
        if len(FILTER_KWARG_MAP[v]) > 0:
            self.filter_kwargs[FILTER_KWARG_MAP[v][0]] = self.k0 
        self._filter()
        self._detect()
        self._regenerate_photos()

    def _set_k1(self, value):
        self.k1 = float(value)
        v = self.filter_type.get()
        if len(FILTER_KWARG_MAP[v]) > 1:
            self.filter_kwargs[FILTER_KWARG_MAP[v][1]] = self.k1 
        self._filter()
        self._detect()
        self._regenerate_photos()

    def _set_k2(self, value):
        self.k2 = float(value)
        v = self.detect_type.get()
        if len(self.detect_kwargs) > 0:
            self.detect_kwargs[DETECT_KWARG_MAP[v][0]] = self.k2 
        self._detect()
        self._regenerate_photos()

    def _set_k3(self, value):
        self.k3 = float(value)
        v = self.detect_type.get()
        if len(self.detect_kwargs) > 1:
            self.detect_kwargs[DETECT_KWARG_MAP[v][1]] = self.k3 
        self._detect()
        self._regenerate_photos()
    
    def _set_k4(self, value):
        self.k4 = float(value)
        v = self.detect_type.get()
        if len(self.detect_kwargs) > 2:
            self.detect_kwargs[DETECT_KWARG_MAP[v][2]] = self.k4 
        self._detect()
        self._regenerate_photos()

    def _set_k5(self, value):
        self.k5 = float(value)
        v = self.detect_type.get()
        if len(self.detect_kwargs) > 3:
            self.detect_kwargs[DETECT_KWARG_MAP[v][3]] = self.k5
        self._detect()
        self._regenerate_photos()

    def _regenerate_primary_photo(self):
        """
        Only regenerate photo_00 corresponding to 
        self.img_00, in the upper left.

        """
        if not self.B0_state:
            self.photo_00 = get_photo(self.img_00, vmax=255.0,
                resize=self.image_resize)
            self.canvas.create_image(
                self.image_pos[0][0],
                self.image_pos[0][1],
                image=self.photo_00,
                anchor=tkinter.NW)
        else:
            positions = label_binary_spots(self.img_11,
                img_int=self.img_01)
            self.photo_00 = get_photo(
                overlay_spots(
                    self.img_00,
                    positions,
                    crosshair_len=self.crosshair_len,
                ),
                vmax=255.0,
                resize=self.image_resize
            )
            self.canvas.create_image(
                self.image_pos[0][0],
                self.image_pos[0][1],
                image=self.photo_00,
                anchor=tkinter.NW)

    def _regenerate_photos(self):
        """
        Update all relevant PIL.ImageTk objects with
        the current value of self.vmax.

        """
        if not self.B0_state:
            self.photo_01 = get_photo(self.img_01, vmax=self.vmax,
                resize=self.image_resize)
            self.photo_10 = get_photo(self.img_10, vmax=self.vmax,
                resize=self.image_resize)
            self.canvas.create_image(
                self.image_pos[1][0], 
                self.image_pos[1][1],
                image=self.photo_01,
                anchor=tkinter.NW)
            self.canvas.create_image(
                self.image_pos[2][0],
                self.image_pos[2][1],
                image=self.photo_10,
                anchor=tkinter.NW)
        else:
            positions = label_binary_spots(
                self.img_11,
                img_int=self.img_01,
            )
            self.photo_00 = get_photo(
                overlay_spots(
                    self.img_00,
                    positions,
                    crosshair_len=self.crosshair_len,
                ),
                vmax=255.0,
                resize=self.image_resize,
            )
            self.photo_01 = get_photo(
                overlay_spots(
                    self.img_01,
                    positions,
                    crosshair_len=self.crosshair_len,
                ),
                vmax=self.vmax,
                resize=self.image_resize
            )
            self.photo_10 = get_photo(
                overlay_spots(
                    self.img_10,
                    positions,
                    crosshair_len=self.crosshair_len,
                ),
                vmax=self.vmax,
                resize=self.image_resize
            )
            self.canvas.create_image(
                self.image_pos[0][0],
                self.image_pos[0][1],
                image=self.photo_00,
                anchor=tkinter.NW)
            self.canvas.create_image(
                self.image_pos[1][0], 
                self.image_pos[1][1],
                image=self.photo_01,
                anchor=tkinter.NW)
            self.canvas.create_image(
                self.image_pos[2][0],
                self.image_pos[2][1],
                image=self.photo_10,
                anchor=tkinter.NW)

        # Don't ever overlay spots on the binary image 
        self.photo_11 = get_photo(self.img_11, vmax=self.vmax,
            resize=self.image_resize)
        self.canvas.create_image(
            self.image_pos[3][0],
            self.image_pos[3][1],
            image=self.photo_11,
            anchor=tkinter.NW)

class MainGUI(object):
    """
    The main GUI. A list of options to launch various sub-
    GUIs.

    """
    def __init__(self, gui_size=200):
        self.gui_size = gui_size

        # The directory to use in file dialogs;
        # initially set to current directory
        self.curr_dir = os.getcwd()

        # Make the main tkinter interface
        self.root = tkinter.Tk()
        self.root.title("quot")

        # Master frame, for organization of buttons
        self.frame = tkinter.Frame(self.root,
            height=self.gui_size, width=self.gui_size)
        self.frame.pack()

        # Main label
        self.L0 = tkinter.Label(self.frame,
            text='quot: SPT detection/tracking optimization',
            font=('Helvetica', 16))
        self.L0.grid(row=0, column=0, pady=10, padx=10)

        # Button formatting kwargs
        button_kwargs = {'activeforeground': '#dadde5',
            'activebackground': '#455462'}
        button_grid_kwargs = {'pady': 10, 'padx': 10,
            'sticky': tkinter.NW}

        # Button to start GUI(), the detection optimizer
        self.B0 = tkinter.Button(self.frame, text='Launch detection optimizer',
            command=self._b0_callback, **button_kwargs)
        self.B0.grid(row=1, column=0, **button_grid_kwargs)

        # Button to launch LocalizeGUI(), the localization
        # interface
        self.B1 = tkinter.Button(self.frame, text='Launch localization',
            command=self._b1_callback, **button_kwargs)
        self.B1.grid(row=2, column=0, **button_grid_kwargs)

        # Button to launch QCGUI(), the QC interface
        self.B2 = tkinter.Button(self.frame, text='Launch spot QC',
            command=self._b2_callback, **button_kwargs)
        self.B2.grid(row=3, column=0, **button_grid_kwargs)

        # Button to launch TrackGUI(), the tracking optimization
        self.B3 = tkinter.Button(self.frame, text='Launch tracking optimizer',
            command=self._b3_callback, **button_kwargs)
        self.B3.grid(row=4, column=0, **button_grid_kwargs)

        # Button to launch BatchTrackGUI(), the equivalent of 
        # LocalizeGUI() for tracking
        self.B4 = tkinter.Button(self.frame, text='Launch batch tracking',
            command=self._b4_callback, **button_kwargs)
        self.B4.grid(row=5, column=0, **button_grid_kwargs)

        # Button to launch analysis module
        self.B5 = tkinter.Button(self.frame, text='Launch analysis',
            command=self._b5_callback, **button_kwargs)
        self.B5.grid(row=6, column=0, **button_grid_kwargs)

        # Button to close the GUI
        self.B6 = tkinter.Button(self.frame, text='Close',
            command=self._b6_callback, **button_kwargs)
        self.B6.grid(row=7, column=0, **button_grid_kwargs)

        # Main loop
        self.root.mainloop()

    def _b0_callback(self, *args):
        """
        Action to execute upon pressing button self.B1.
        Opens a file dialog for the user to select a file
        or directory for detection optimization.

        """
        # Prompt the user to enter a file
        self.curr_file = filedialog.askopenfilename(
            initialdir=self.curr_dir, 
            title='Select file for detection optimization',
        )

        # Try to open the file
        reader = qio.ImageFileReader(self.curr_file)

        # Save the parent directory of this file for
        # any future dialogs
        self.curr_dir = os.path.dirname(self.curr_file)

        # Prompt the user to input the subregion of interest
        n_frames, N, M = reader.get_shape()
        defaults = {
            'Start frame': 0,
            'Stop frame': min([100, n_frames]),
            'Lower y limit': 0,
            'Upper y limit': N,
            'Lower x limit': 0,
            'Upper x limit': M, 
        }
        choices = VariableEntryPrompt(defaults,
            title='Choose subregion')
        subregion = [
            [choices['Lower y limit'],
            choices['Upper y limit']],
            [choices['Lower x limit'],
            choices['Upper x limit']],
        ]
        frame_limits = [choices['Start frame'],
            choices['Stop frame']]

        # Close the temporary reader
        reader.close()

        # Start a new Toplevel object for the GUI subwindow
        self.top = tkinter.Toplevel()

        # Launch the GUI with the file
        GUI(self.curr_file, root=self.top,
            subregion=subregion, frame_limits=frame_limits)

    def _b1_callback(self, *args):
        """
        Action to execute upon pressing button self.B1.
        Opens a file dialog for the user to select a file
        or directory for localization.

        """
        # Start a new Toplevel object for the GUI subwindow
        self.top = tkinter.Toplevel()

        # Open the localization GUI 
        LocalizeGUI(root=self.top, curr_dir=self.curr_dir)

    def _b2_callback(self, *args):
        """
        Action to execute upon pressing button self.B2.
        Opens a dialog for the user to run quality control
        actions on localized spots.

        """
        # Start a new Toplevel object for the GUI subwindow
        self.top = tkinter.Toplevel()

        # Open the QC GUI
        QCGUI(root=self.top, curr_dir=self.curr_dir)

    def _b3_callback(self, *args):
        """
        Action to execute upon pressing button self.B3.
        Opens a dialog for the user to run tracking
        optimization.

        """
        # Prompt the user to enter a file
        loc_file = filedialog.askopenfilename(
            initialdir=self.curr_dir, 
            title='Select *locs.csv file',
        )
        self.curr_dir = os.path.dirname(loc_file)

        # Try to find the corresponding image file
        nd2_file = loc_file.replace('_locs.csv', '.nd2')
        tif_file = loc_file.replace('_locs.csv', '.tif')
        if os.path.isfile(nd2_file):
            image_file = nd2_file
        elif os.path.isfile(tif_file):
            image_file = tif_file
        else:
            image_file = filedialog.askopenfilename(
                initialdir=self.curr_dir,
                title='Select image file',
            )

        # Try to open the file
        self.curr_file = image_file
        reader = qio.ImageFileReader(self.curr_file)

        # Save the parent directory of this file for
        # any future dialogs
        self.curr_dir = os.path.dirname(self.curr_file)

        # Prompt the user to input the subregion of interest
        n_frames, N, M = reader.get_shape()
        defaults = {
            'Start frame': 0,
            'Stop frame': min([100, n_frames]),
            'Lower y limit': 0,
            'Upper y limit': N,
            'Lower x limit': 0,
            'Upper x limit': M, 
            'Upsampling': 2,
            'Frame interval (s)': 0.00548,
            'Pixel size (um)': 0.16,
        }
        choices = VariableEntryPrompt(defaults,
            title='Choose subregion')
        subregion = [
            [choices['Lower y limit'],
            choices['Upper y limit']],
            [choices['Lower x limit'],
            choices['Upper x limit']],
        ]
        frame_limits = [choices['Start frame'],
            choices['Stop frame']]

        # Close the reader
        reader.close()

        # Start a new Toplevel object for the GUI subwindow
        self.top = tkinter.Toplevel()

        # Launch TrackGUI
        TrackGUI(loc_file, image_file, root=self.top,
            subregion=subregion, frame_limits=frame_limits,
            upsample=int(choices['Upsampling']),
            pixel_size_um=float(choices['Pixel size (um)']),
            frame_interval_sec=float(choices['Frame interval (s)']))

    def _b4_callback(self, *args):
        """
        Launch BatchTrackGUI, for tracking multiple
        localization files.

        NOT YET IMPLEMENTED.

        """
        pass 

    def _b5_callback(self, *args):
        """
        Launch an analysis module.

        NOT YET IMPLEMENTED.

        """
        pass 

    def _b6_callback(self, *args):
        """
        Response to button B6: close the GUI.

        Due to a weird entanglement with matplotlib,
        must also close matplotlib if the user has
        generated any plots while the GUI is open.

        """
        plt.close()
        self.root.destroy()

class LocalizeGUI(object):
    """
    GUI for the user to run localization on a file or 
    set of files.

    init
    ----
        root : either None, if launching as a standalone GUI,
            or a tkinter.Toplevel object if launching from
            another tkinter GUI

    """
    def __init__(self, root=None, curr_dir=None, gui_size=200):
        self.gui_size = gui_size 

        # The current configuration settings for 
        # filtering, detection, and localization, 
        # by default set to nothing
        self.configs = {}

        # Instantiate the main tkinter window, creating
        # a root if there are no parent GUIs
        if root is None:
            self.root = tkinter.Tk()
        else:
            self.root = root 
        self.root.title("Localization")

        # The directory that menus start in 
        if curr_dir is None:
            self.curr_dir = os.getcwd()
        else:
            self.curr_dir = curr_dir 

        # Format kwargs
        button_kwargs = {'activeforeground': '#dadde5',
            'activebackground': '#455462'}
        button_grid_kwargs = {'pady': 5, 'padx': 5,
            'sticky': tkinter.NW}
        label_kwargs = {'sticky': tkinter.NW, 'pady': 5,
            'padx': 5}

        # Frame for organizing widgets
        self.frame = tkinter.Frame(self.root,
            height=self.gui_size, width=self.gui_size)
        self.frame.pack()

        # Label L0: Section 1
        self.L0 = tkinter.Label(self.frame,
            text='1. ND2/TIF files for localization',
            font=('Helvetica', 16))
        self.L0.grid(row=0, column=0, **label_kwargs)

        # Button B0: select files
        self.B0 = tkinter.Button(self.frame, text='Select file',
            command=self._b0_callback, **button_kwargs)
        self.B0.grid(row=1, column=0, **button_grid_kwargs)

        # Button B1: select directory
        self.B1 = tkinter.Button(self.frame, text='Select directory',
            command=self._b1_callback, **button_kwargs)
        self.B1.grid(row=2, column=0, **button_grid_kwargs)

        # Label L1: show currently selected file/directory
        self.target_path = tkinter.StringVar(self.root)
        self.target_path.set("None")
        self.L1 = tkinter.Label(self.frame,
            textvariable=self.target_path)
        self.L1.grid(row=3, column=0, **label_kwargs)

        # Label L2: Section 2 heading
        self.L2 = tkinter.Label(self.frame,
            text='2. Filtering/detection settings',
            font=('Helvetica', 16))
        self.L2.grid(row=4, column=0, **label_kwargs)

        # Button B2: select settings .yaml file
        self.B2 = tkinter.Button(self.frame, text='Select .yaml file',
            command=self._b2_callback, **button_kwargs)
        self.B2.grid(row=5, column=0, **button_grid_kwargs)

        # Label L3: show currently selected settings file
        self.config_path = tkinter.StringVar(self.root)
        self.config_path.set("None")
        self.L3 = tkinter.Label(self.frame, 
            textvariable=self.config_path)
        self.L3.grid(row=6, column=0, **label_kwargs)

        # Label L4: Section 3 heading
        self.L4 = tkinter.Label(self.frame, 
            text='3. Subpixel localization algorithm',
            font=('Helvetica', 16))
        self.L4.grid(row=8, column=0, **label_kwargs)

        # Menu M0: Localization algorithm options
        self.loc_method = tkinter.StringVar(self.root)
        self.loc_method.set("None")
        method_choices = LOCALIZE_METHODS.keys()
        self.M0 = tkinter.OptionMenu(self.frame, self.loc_method,
            *method_choices)
        self.M0.grid(row=9, column=0, pady=5, padx=5, 
            sticky=tkinter.NW)

        # Button B3: modify localization settings,
        # launching an option window specific to the 
        # GUI type
        self.B3 = tkinter.Button(self.frame, text='Parameters',
            command=self._b3_callback, **button_kwargs)
        self.B3.grid(row=10, column=0, **button_grid_kwargs)

        # Label L5: Section 4 heading
        self.L5 = tkinter.Label(self.frame,
            text='4. Run', font=('Helvetica', 16))
        self.L5.grid(row=11, column=0, **label_kwargs)

        # Button B4: launch localization
        self.B4 = tkinter.Button(self.frame, text='Run',
            command=self._b4_callback, **button_kwargs)
        self.B4.grid(row=12, column=0, **button_grid_kwargs)

        # Label L6: show whether the current settings
        # are valid
        self.L6_var = tkinter.StringVar()
        self.L6_var.set("")
        self.L6 = tkinter.Label(self.frame, 
            textvariable=self.L6_var,
            justify=tkinter.LEFT)
        self.L6.grid(row=13, column=0, 
            **label_kwargs)

        # Start the main tkinter loop
        self.root.mainloop()

    def _settings_valid(self):
        """
        Check whether the currently selected localization
        settings are valid.

        returns
        -------
            (
                bool, whether settings are valid;
                str, an error message 
            )

        """
        target_file = self.target_path.get()
        config_file = self.config_path.get()

        # Basic checks
        if target_file == "None":
            e = "No target files selected"
            return False, e
        if config_file == "None":
            e = "No settings file selected"
            return False, e

        # Check that paths are valid
        if not qio.path_exists(target_file):
            e = "Target filepath %s does not exist" % target_file
            return False, e

        if not qio.path_exists(config_file):
            e = "Config filepath %s does not exist" % config_file
            return False, e

        # Check current config settings
        configs = self.configs 
        sections = [str(j) for j in configs.keys()]

        # Check that the configs have the right subsections
        if 'filtering' not in sections:
            e = "Config file %s does not contain a " \
                "filtering subsection" % config_file
            return False, e  
        else:
            filter_configs = configs['filtering']

        if 'detection' not in sections:
            e = "Config file %s does not contain a " \
                "detection section" % config_file
            return False, e
        else:
            detect_configs = configs['detection']

        if ('localization' not in sections):
            if hasattr(self, 'loc_configs'):
                loc_configs = self.loc_configs 
            else:
                e = "No localization method specified"
                return False, e 
        else:
            loc_configs = configs['localization']

        # Make sure each config section is dict
        try:
            assert isinstance(filter_configs, dict)
            assert isinstance(detect_configs, dict)
            assert isinstance(loc_configs, dict)
        except AssertionError:
            e = "One of the config settings is not a dict"
            return False, e

        # Make sure that each config section species
        # a method key 
        try:
            assert 'method' in filter_configs.keys()
        except AssertionError:
            e = "No filtering method specified"
            return False, e 
        try:
            assert 'method' in detect_configs.keys()
        except AssertionError:
            e = "No detection method specified"
            return False, e
        try:
            assert 'method' in loc_configs.keys()
        except AssertionError:
            e = "No localization method specified"
            return False, e 

        filter_method = filter_configs['method']
        detect_method = detect_configs['method']
        loc_method = loc_configs['method']

        # Check that filtering settings are valid
        for k in filter_configs.keys():
            if (str(k) != 'method') and (str(k) != 'block_size') \
                and (str(k) not in FILTER_KWARG_MAP[filter_method]):
                e = "Unknown filtering kwargs '%s'" % str(k)
                return False, e

        # Check that detection settings are valid
        for k in detect_configs.keys():
            if (str(k) != 'method') and (str(k) not in \
                DETECT_KWARG_MAP[detect_method]):
                e = "Unknown detection kwarg '%s'" % str(k)
                return False, e

        # Check that localization settings are valid
        for k in loc_configs.keys():
            if (str(k) != 'method') and (k not in \
                LOCALIZE_KWARGS_DEFAULT[loc_method].keys()):
                e = "Unknown localization kwarg '%s'" % str(k)
                return False, e

        # If we pass all tests 
        return True, "Settings valid"

    def _show_settings(self, *args):
        """
        Show the current configuration settings
        to the user.

        """
        configs = self.configs
        out = "\nCurrent:\n"
        for k0 in configs.keys():
            out += "%s:\n" % str(k0)
            for k1 in configs[k0].keys():
                out += "\t%s:\t%r\n" % (str(k1),
                    configs[k0][k1])
        out += "\n"
        self.L6_var.set(out)

    def _b0_callback(self, *args):
        self.curr_file = filedialog.askopenfilename(
            initialdir=self.curr_dir,
            title='Select file/directory'
        )
        if len(self.curr_file) == 0:
            self.L6_var.set("No file selected")
            return 

        self.curr_dir = os.path.dirname(self.curr_file)
        self.target_path.set(self.curr_file)

    def _b1_callback(self, *args):
        """
        Callback for pressing button self.B1, which
        prompts the user to enter a directory containing
        ND2 or TIF files for analysis.

        """
        self.curr_file = filedialog.askdirectory(
            initialdir=self.curr_dir,
            title='Select .yaml settings',
        )
        if len(self.curr_file) == 0:
            self.L6_var.set("No directory selected")
            return 

        self.curr_dir = os.path.dirname(self.curr_file)
        self.target_path.set(self.curr_file)

    def _b2_callback(self, *args):
        """
        Callback for pressing button self.B2, which
        prompts the user to select a .yaml 
        localization settings file.

        """
        # Prompt user to select file
        self.curr_file = filedialog.askopenfilename(
            initialdir=self.curr_dir,
            title='Select settings file',
            filetypes=(("yaml files", "*.yaml"), ("all files", "*.*")),
        )

        if len(self.curr_file) == 0:
            self.L6_var.set("No settings file selected")
            return 

        # Save filepath
        self.curr_dir = os.path.dirname(self.curr_file)
        self.config_path.set(self.curr_file)

        # Try to read the file
        try:
            self.configs = qio.read_config(self.config_path.get())

            # Look for localization keywords in particular,
            # to set defaults for self.B3 
            if 'localization' in self.configs.keys():
                self.loc_configs = self.configs['localization']
                if 'method' in self.loc_configs.keys():
                    self.loc_method.set(self.loc_configs['method'])
            else:
                self.loc_method.set('None')

            # Show the settings to the user 
            self._show_settings()
        except FileNotFoundError:
            self.L6_var.set("Settings file not found")

    def _b3_callback(self, *args):
        """
        Callback for pressing button self.B3, which
        prompts the user to modify the localization
        settings.

        """
        # Get the current localization method
        method = self.loc_method.get()

        # Make sure this method exists
        assert method in LOCALIZE_METHODS.keys()

        # Prompt the user to modify the localization
        # settings
        self.loc_configs = VariableEntryPrompt(
            LOCALIZE_KWARGS_DEFAULT[method],
            title='%s parameters' % method)

        # Also record the method name itself
        self.loc_configs['method'] = method 

        # Modify the present configuration settings
        self.configs['localization'] = self.loc_configs

        # Show the result to the user 
        self._show_settings()

    def _b4_callback(self, *args):
        """
        Callback for pressing button self.B4, which
        runs localization with the selected settings
        on the selected input files.

        """
        self._show_settings()

        # Check that the current settings are valid
        valid, error_message = self._settings_valid()

        # Show the current error message
        self.L6_var.set(error_message)

        # Try to run with the present settings
        if valid:
            path = self.target_path.get()

            # Format list of input paths
            if os.path.isdir(path):
                target_files = glob("%s/*.nd2" % path) + \
                    glob("%s/*.tif" % path)
            elif os.path.isfile(path):
                target_files = [path]

            # Run detection only
            if self.configs['localization']['method'] == 'detection_only':
                sub_configs = {i: self.configs[i] for i in self.configs.keys() \
                    if i != 'localization'}
                self.L6_var.set("Running detection only...")
                localize._loc_files(target_files, verbose=True,
                    **sub_configs)

            # Run localization
            else:
                self.L6_var.set("Localizing...")
                localize._loc_files(target_files, verbose=True,
                    **self.configs)
                self.L6_var.set("Finished localizing %s" % path)

class QCGUI(object):
    """
    A GUI with localization quality control features.

    Features:

        1. Select and open a *locs.csv file, and show
            the distribution of localizations in this
            file

        2. For a set of localizations with several 
            attributes, show a 2D histogram of two
            of the attributes across the whole set
            of localizations

        3. Allow the user to set thresholds on 
            localizations according to specific
            attributes

        4. Window with the actual movie, overlaying
            localizations and showing green if they
            pass a threshold and red otherwise

        5. Allow the user to draw a mask on localizations

    Attributes:

        self.loc_file : str, path to localization file
        self.image_file : str, path to the image file.
            When self.loc_file is set, tries to find this
            by looking for similar names in the same
            directory
        self.LOC_ATTRIBS, the localization attributes,
            set by the columns in self.loc_file

        self.locs : the pandas.DataFrame with raw localizations
            for the whole movie (shape (n_locs, n_attrib))
        self.positions : a 2D ndarray with the YX 
            coordinates for the frame range that is currently
            selected, shape (n_positions, 2)
        self.condition : a 1D ndarray (dtype bool) that indicates
            whether each of the localizations in *self.positions*
            passes the current QC filters. shape (n_positions,)

    """
    def __init__(self, root=None, curr_dir=None,
        gui_size=300, upsampling=2):

        ## ATTRIBUTES

        # How much to upsample the images
        self.upsampling = upsampling 

        # Size of a subwindow
        self.gui_size = gui_size 

        # Currently selected localization file
        self.loc_file = tkinter.StringVar()

        # The list of columns in this localization file
        self.loc_attribs = ['something']

        # All localizations in the movie
        self.all_locs = None 

        # Localizations in the current frame range
        self.locs = None 

        # The (frame_idx, y0, x0) tuples for those 
        # localizations
        self.positions = None 

        # Currently selected image file
        self.image_file = tkinter.StringVar()

        # The number of frames in this image file
        self.n_frames = 100

        # The height and width of the image file
        self.IH = 100
        self.IW = 100

        # The height and width of the rescaled image file
        self.img_shape = np.array([self.gui_size, self.gui_size])

        # Current file reader, corresponding to 
        # self.image_file
        self.reader = None 

        # The current frame index in the movie
        self.frame_index = tkinter.IntVar()

        # The raw image corresponding to this frame
        # index
        self.raw_img = None 

        # The (potentially filtered/overlaid) image
        # corresponding to this frame index
        self.img = None 

        # The PIL.ImageTk object corresponding to self.img
        self.photo = None 

        # Whether we are currently overlaying localizations
        # on the raw movie
        self.overlay_locs = False 

        # The frame limits. These determine the limits
        # of the frame index slider, and also the
        # list of localizations that are subjected to
        # filtering steps
        self.frame_limits = (0, 100)

        # vmin/vmax for image representation
        self.vmin = 0
        self.vmax = 5000

        # A string that is displayed to the user, giving
        # feedback
        self.user_info = tkinter.StringVar()


        ## FORMATTING 
        label_kwargs = {'font': ('Helvetica', 16)}
        button_kwargs = {'activeforeground': '#dadde5',
            'activebackground': '#455462'}
        slider_kwargs = {'orient': tkinter.HORIZONTAL,
            'activebackground': '#000000', 'resolution': 1,
            'length': 200, 'font': ('Helvetica', '16'),
            'sliderrelief': tkinter.FLAT, 'bd': 0}
        grid_kwargs = {'sticky': tkinter.NW, 'pady': 5,
            'padx': 5}
        self.grid_kwargs = grid_kwargs 
    

        ## INTERFACE LAYOUT
        if root is None:
            root = tkinter.Tk()
        self.root = root 
        self.root.title("Localization QC")

        # Default directory for file dialogs
        if curr_dir is None:
            curr_dir = os.getcwd()
        self.curr_dir = curr_dir 

        # Left frame, for buttons etc.
        self.left_frame = tkinter.Frame(self.root,
            height=self.gui_size, width=self.gui_size)
        self.left_frame.grid(row=0, column=0)

        # Right frame, for showing raw image 
        self.right_frame = tkinter.Frame(self.root,
            height=self.gui_size, width=self.gui_size)
        self.right_frame.grid(row=0, column=1)


        # LEFT FRAME

        # Label L0: Section header for file selection
        self.L0 = tkinter.Label(self.left_frame, 
            text="Select files", **label_kwargs)
        self.L0.grid(row=0, column=0, **grid_kwargs)

        # Button B0: Select localization file
        self.B0 = tkinter.Button(self.left_frame,
            text="Select *locs.csv file",
            command=self._B0_callback,
            **button_kwargs)
        self.B0.grid(row=1, column=0, **grid_kwargs)

        # Label LB0: Says what the currently selected loc
        # file is
        self.LB0 = tkinter.Label(self.left_frame,
            textvariable=self.loc_file)
        self.LB0.grid(row=2, column=0, **grid_kwargs)

        # Button B1: Select image file 
        self.B1 = tkinter.Button(self.left_frame,
            text='Select *.tif or *.nd2 file',
            command=self._B1_callback,
            **button_kwargs)
        self.B1.grid(row=3, column=0, **grid_kwargs)

        # Label LB1: Show the currently selected image
        # file
        self.LB1 = tkinter.Label(self.left_frame,
            textvariable=self.image_file)
        self.LB1.grid(row=4, column=0, **grid_kwargs)

        # Button B2: Generate localization summary
        self.B2 = tkinter.Button(self.left_frame, 
            text='Generate localization summary',
            command=self._B2_callback)
        self.B2.grid(row=5, column=0, **grid_kwargs)

        # Label L1: Section header for localization filtering
        self.L1 = tkinter.Label(self.left_frame, 
            text="Spatial masking", **label_kwargs)
        self.L1.grid(row=6, column=0, **grid_kwargs)

        # Button B3: Draw a mask on the localization density
        self.B3 = tkinter.Button(self.left_frame,
            text='Draw mask', command=self._B3_callback)
        self.B3.grid(row=7, column=0, **grid_kwargs)

        # Label L4: Section header for attribute masking
        self.L3 = tkinter.Label(self.left_frame,
            text='Attribute masking', **label_kwargs)
        self.L3.grid(row=8, column=0, **grid_kwargs)

        # Button B5: Draw a mask on the attributes
        self.B5 = tkinter.Button(self.left_frame,
            text='Draw mask', command=self._B5_callback)
        self.B5.grid(row=9, column=0, **grid_kwargs)

        # Label L3: Section header for 2D histogram
        self.L3 = tkinter.Label(self.left_frame,
            text='Attribute 2D histogram', **label_kwargs)
        self.L3.grid(row=11, column=0, **grid_kwargs)

        # Button B4: 2D histogram of attributes
        self.B4 = tkinter.Button(self.left_frame,
            text='Draw histogram', command=self._B4_callback)
        self.B4.grid(row=14, column=0, **grid_kwargs)

        # Menu M0: The first attribute for the histogram
        # produced by button B4
        self.B4_attrib_0 = tkinter.StringVar()
        self.M0 = tkinter.OptionMenu(self.left_frame,
            self.B4_attrib_0, *self.loc_attribs)
        self.M0.grid(row=12, column=0, **grid_kwargs)

        # Menu M1: The second attribute for the histogram
        # produced by button B4
        self.B4_attrib_1 = tkinter.StringVar()
        self.M1 = tkinter.OptionMenu(self.left_frame,
            self.B4_attrib_1, *self.loc_attribs)
        self.M1.grid(row=13, column=0, **grid_kwargs)

        # Label L4: Section header for applying a mask
        self.L4 = tkinter.Label(self.left_frame, 
            text='Filter localizations', **label_kwargs)
        self.L4.grid(row=15, column=0, **grid_kwargs)

        # Button B6: Launch a window that asks the user
        # which columns to apply
        self.B6 = tkinter.Button(self.left_frame,
            text='Apply filters', command=self._B6_callback)
        self.B6.grid(row=16, column=0, **grid_kwargs)

        # Button B7: Save the current filter settings to 
        # the locs file
        self.B7 = tkinter.Button(self.left_frame, 
            text='Save filter info to file', command=self._B7_callback)
        self.B7.grid(row=17, column=0, **grid_kwargs)

        # Label L2: A flexible user feedback label that
        # tracks self.user_info
        self.L2 = tkinter.Label(self.left_frame,
            textvariable=self.user_info)
        self.L2.grid(row=18, column=0, **grid_kwargs)


        # RIGHT FRAME

        # Canvas, for displaying images 
        self.canvas = tkinter.Canvas(self.right_frame,
            height=self.gui_size, width=self.gui_size)
        self.canvas.grid(row=0, column=0, **grid_kwargs)

        # Slider S0, for changing the frame index 
        self.S0 = tkinter.Scale(self.right_frame,
            from_=self.frame_limits[0], to=self.frame_limits[1]-1,
            command=self._S0_callback, label='Frame',
            **slider_kwargs)
        self.S0.grid(row=1, column=0, **grid_kwargs)
        self.S0.set(0)

        # Slider S1, for adjusting vmax 
        self.S1 = tkinter.Scale(self.right_frame,
            from_=0, to=5000, command=self._S1_callback,
            label='LUT max', **slider_kwargs)
        self.S1.grid(row=2, column=0, **grid_kwargs)
        self.S1.set(5000)

        # Slider S2, for adjusting vmin
        self.S2 = tkinter.Scale(self.right_frame,
            from_=0, to=5000, command=self._S2_callback,
            label='LUT min', **slider_kwargs)
        self.S2.grid(row=3, column=0, **grid_kwargs)
        self.S2.set(0)

        # Button BR0, 
        self.BR0 = tkinter.Button(self.right_frame,
            text='Set frame range', command=self._BR0_callback)
        self.BR0.grid(row=4, column=0, **grid_kwargs)

        # Button BR1, for toggling localization overlay
        self.BR1 = tkinter.Button(self.right_frame,
            text='Overlay locs', command=self._BR1_callback)
        self.BR1.grid(row=5, column=0, **grid_kwargs)


        # Run the GUI
        self.root.mainloop()

    def _B0_callback(self, *args):
        """
        Execute upon pressing button self.B0, which
        selects a localization file.

        """
        self._select_loc_file()

    def _B1_callback(self, *args):
        """
        Execute upon pressing button self.B1, which
        selects an image file.

        """
        self._select_image_file()

    def _B2_callback(self, *args):
        """
        Execute upon pressing button self.B2, which 
        generates a localization summary.

        """
        visualize.localization_summary(
            pd.read_csv(self.loc_file.get()),
            self.image_file.get(),
            out_png='%s_localization_summary.png' % \
                self.loc_file.get().replace('.csv', '')
        )

    def _B3_callback(self, *args):
        """
        Execute upon pressing button self.B3, which 
        prompts the user to draw a manual mask.

        """
        # Ask the user whether they want to 
        # draw on the raw image or on localization density
        choice = OptionPrompt(['Max intensity projection',
            'Localization density'], title='Mask type',
            label='Draw the mask on:')

        # If localization density, make a KDE for the
        # localizations
        if choice == 'Localization density':
            up_factor = 10 
            img = visualize.loc_density(self.all_locs,
                self.IH, self.IW, plot=False, pixel_size_um=1.0,
                upsampling_factor=up_factor, kernel_width=1.0)

            # Prompt the user to draw mask 
            self.mask_edge = np.asarray(DrawGUI(img).path) / up_factor

        elif choice == 'Max intensity projection':
            img = self.reader.max_int_proj()

            # Prompt the user to draw mask 
            self.mask_edge = np.asarray(DrawGUI(img).path)

        # Format path as a matplotlib.path.Path object
        self.mask_path = Path(self.mask_edge, closed=True)

        # Run masking on all of the localizations
        self.all_locs['in_mask'] = self.mask_path.contains_points(
            np.asarray(self.all_locs[['y0', 'x0']]))

        # Print to terminal
        print('Inside mask: ', self.all_locs['in_mask'].sum())
        print('Outside mask: ', (~self.all_locs['in_mask']).sum())

        # Make masking summary plot
        visualize.mask_summary(self.mask_path, self.all_locs,
            self.reader.get_shape()[1], self.reader.get_shape()[2],
            out_png='%s_masking_summary.png' % self.loc_file.get().replace('.csv', ''))

    def _B4_callback(self, *args):
        """
        Execute upon pressing button self.B4, which 
        produces a 2D histogram of two attributes.

        """
        # Get the current values of menus M0 and M1 
        c0 = self.B4_attrib_0.get()
        c1 = self.B4_attrib_1.get()

        # Make the plot
        visualize.attrib_dist_2d(self.all_locs, 
            c0, c1)

    def _B5_callback(self, *args):
        """
        Execute upon pressing button self.B5, which
        prompts the user to draw a mask on two 
        attributes.

        """
        # Get the user to input which attributes
        # they want to draw on 
        cols = self.all_locs.columns
        defaults = {'Attribute 1': cols[0],
            'Attribute 2': cols[1],
            'Number of bins in each axis': 150}
        choices = VariableEntryPrompt(defaults, 
            title='Choose attributes')

        # Make a histogram of these attributes
        attrib_0 = choices['Attribute 1']
        attrib_1 = choices['Attribute 2']
        n_bins = choices['Number of bins in each axis']
        H, edges_0, edges_1 = utils.attrib_histogram_2d(
            self.all_locs, attrib_0, attrib_1, n_bins=n_bins)

        # Bin sizes along each dimension
        bin_size_0 = edges_0[1] - edges_0[0]
        bin_size_1 = edges_1[1] - edges_1[0]

        # Prompt the user to draw a mask 
        self.attrib_mask_edge = DrawGUI(H,
            axis_labels_yx=(attrib_0, attrib_1)).path

        # Convert to native units
        mask_edge = np.zeros(self.attrib_mask_edge.shape)
        mask_edge[:,0] = self.attrib_mask_edge[:,1]* bin_size_0 + edges_0[0]
        mask_edge[:,1] = self.attrib_mask_edge[:,0]* bin_size_1 + edges_1[0]
        self.attrib_mask_edge = mask_edge 

        # Format as matplotlib.path.Path 
        self.attrib_mask_path = Path(self.attrib_mask_edge, closed=True)

        # Assign a new column in the original dataframe
        self.all_locs['in_attribute_mask'] = self.attrib_mask_path.contains_points(
            np.asarray(self.all_locs[[attrib_0, attrib_1]]))

        # Show the result
        visualize.attribute_mask_summary(
            self.attrib_mask_path,
            self.all_locs,
            edges_0,
            edges_1,
            attrib_0, 
            attrib_1,
            xlabel=attrib_0,
            ylabel=attrib_1,
            out_png = '%s_attribute_mask_summary.png' % self.loc_file.get().replace('.csv', '')
        )

    def _B6_callback(self, *args):
        """
        Execute upon pressing button B6, which prompts
        the user to select columns for filtering the 
        localizations.

        """
        # Get the list of columns in the dataframe
        # that are Boolean
        bool_cols = [c for c in self.all_locs.columns \
            if self.all_locs[c].dtype == 'bool']

        # Also give the user the option to apply no
        # filter
        bool_cols = ["None"] + bool_cols 

        # Prompt the user to select three different
        # conditions on the localizations
        choices = OptionsPrompt([bool_cols, bool_cols, 
            bool_cols], ['Filter 1', 'Filter 2', 'Filter 3'])

        # Exclude choices that are None 
        choices = [i for i in choices if i != "None"]

        # Take the condition
        self.all_condition = np.asarray(self.all_locs[choices]).all(axis=1)
        
        # Update self.condition
        self.condition = self.all_condition[self.all_locs['frame_idx'].isin(
            np.arange(self.frame_limits[0], self.frame_limits[1]))]

        # Update the frame
        self._set_frame()

    def _B7_callback(self, *args):
        """
        Executes upon pressing button B7, which saves
        the current filtering info (including any masks)
        to a file.

        The current filtering condition, which may be a 
        combination of several filtering conditions, 
        is saved as a column `in_filter`.

        """
        if isinstance(self.all_locs, pd.DataFrame):
            self.all_locs['in_filter'] = self.all_condition
            self.all_locs.to_csv(self.loc_file.get())
            self.user_info.set("Saved filtering info to %s" % self.loc_file.get())
        else:
            self.user_info.set("No dataframe loaded")

    def _S0_callback(self, frame_index):
        """
        Execute upon changing slider S0, which 
        sets the frame index.

        """
        self._set_frame(frame_index=int(frame_index))

    def _S1_callback(self, vmax):
        """
        Execute upon changing slider S1, which 
        sets image vmax.

        """
        self._set_vmin_vmax(vmax=float(vmax))

    def _S2_callback(self, vmin):
        """
        Execute upon changing slider S2, which 
        sets image vmin.

        """
        self._set_vmin_vmax(vmin=float(vmin))

    def _BR0_callback(self, *args):
        """
        Execute upon pressing button BR0, which 
        prompts the user to change the frame limits.

        """
        # Prompt the user to change the frame limits
        defaults = {'Start frame': self.frame_limits[0],
            'Last frame': self.frame_limits[1]}
        result = VariableEntryPrompt(defaults)

        # Set the frame limits
        self._set_frame_limits(
            start_frame=result['Start frame'],
            stop_frame=result['Stop frame']
        )

    def _BR1_callback(self, *args):
        """
        Execute upon pressing button BR1, which 
        toggles whether localizations are overlaid
        onto the raw image.

        """
        # If no localizations, do nothing
        if self.all_locs is None:
            return 

        # Change the state of the switch 
        self.overlay_locs = not self.overlay_locs

        # Reload the present frame, with overlay
        self._set_frame()

    def _select_loc_file(self):
        """
        Prompt the user to select a localization file,
        then load it.

        """
        # File selector
        loc_file = filedialog.askopenfilename(
            initialdir=self.curr_dir,
            title='Select localization file',
            filetypes = (("csv files", "*.csv"), 
                ("all files", "*.*"))
        )

        # Load this loc file
        self._load_loc_file(loc_file)

    def _load_loc_file(self, loc_file):
        """
        Load a localization CSV into memory.

        Also try to find the corresponding image file
        and load it if possible.

        args
        ----
            loc_file : str, path to file

        """

        # Filter out bad files
        if len(loc_file) == 0 or not os.path.isfile(loc_file):
            self.user_info.set("Bad localization file: %s" % loc_file)
            return 

        self.loc_file.set(loc_file)
        self.curr_dir = os.path.dirname(loc_file)

        # Try to open this file
        self.all_locs = pd.read_csv(loc_file)
        self.loc_attribs = self.all_locs.columns 

        # Try to find the corresponding image file
        tif_file = loc_file.replace('_locs.csv', '.tif')
        nd2_file = loc_file.replace('_locs.csv', '.nd2')
        if os.path.isfile(tif_file):
            self._load_image_file(tif_file)
        elif os.path.isfile(nd2_file):
            self._load_image_file(nd2_file)

        # Get all of the localizations in the current
        # frame range
        self.locs = self.all_locs[self.all_locs['frame_idx'].isin(
            np.arange(self.frame_limits[0], self.frame_limits[1]))]

        # Save the positions of these localizations as 
        # ndarray for speed
        self.positions = np.asarray(self.locs[['frame_idx', 'y0', 'x0']])

        # Set the current filtering condition
        self.all_condition = np.ones(len(self.all_locs), dtype='bool')
        self.condition = self.all_condition[self.all_locs['frame_idx'].isin(
            np.arange(self.frame_limits[0], self.frame_limits[1]))]

        # Update menus M0 and M1, which should produce
        # options from the columns of self.all_locs
        self.loc_attribs = self.all_locs.columns
        self.M0.destroy()
        self.M1.destroy()
        self.B4_attrib_0.set(self.loc_attribs[0])
        self.M0 = tkinter.OptionMenu(self.left_frame,
            self.B4_attrib_0, *self.loc_attribs)
        self.M0.grid(row=12, column=0, **self.grid_kwargs)
        self.B4_attrib_1.set(self.loc_attribs[1])
        self.M1 = tkinter.OptionMenu(self.left_frame,
            self.B4_attrib_1, *self.loc_attribs)
        self.M1.grid(row=13, column=0, **self.grid_kwargs)

    def _select_image_file(self):
        """
        Prompt the user to select an image file,
        then load it.

        """
        # File selector
        image_file = filedialog.askopenfilename(
            initialdir=self.curr_dir,
            title='Select image file',
            filetypes = (("ND2 files", "*.nd2"), 
                ("TIF files", "*.tif"),
                ("all files", "*.*"))
        )

        # Load the file
        self._load_image_file(image_file)

    def _load_image_file(self, image_file):
        """
        Try to load an image file into memory.

        args
        ----
            image_file : str, path 

        """
        # Make sure path exists
        if len(image_file)==0 or not os.path.isfile(image_file):
            self.user_info.set("Bad path: %s" % image_file)
            return 

        # Save the name of the image file
        self.image_file.set(image_file)

        # Make a file reader 
        self.reader = qio.ImageFileReader(image_file)

        # Get the image shape
        self.n_frames, self.IH, self.IW = self.reader.get_shape()

        # Image resizing factor
        self.resize = min([self.gui_size/(self.IH),
            self.gui_size/(self.IW)])
        self.img_shape = np.array([
            self.IH*self.resize,
            self.IW*self.resize,
        ]).astype('uint16')

        # Show the current image in the right frame
        self._set_frame(self.frame_index.get())

        # Get default vmin and vmax from the selected
        # frame range
        self.vmin, self.vmax = self.reader.min_max(
            self.frame_limits[0], self.frame_limits[1])

        # Adjust the vmax/vmin sliders according
        self._set_vmin_vmax(vmax=self.vmax, vmin=self.vmin)

        # Reset the frame index slider if necessary
        if self.n_frames < self.frame_limits[1]:
            self._set_frame_limits(stop_frame=self.n_frames)

    def _set_frame_limits(self, start_frame=None, stop_frame=None):
        """
        Change the start or stop frame for the 
        interval of the movie currently under
        observation.

        """
        # Adjust self.frame_limits
        if not (start_frame is None):
            self.frame_limits[0] = start_frame 
        if not (stop_frame is None):
            self.frame_limits[1] = min([stop_frame, self.n_frames])

        # If the current frame index is outside
        # of the range, reset it
        if self.frame_index.get() < self.frame_limits[0] or \
            self.frame_index.get() >= self.frame_limits[1]:
            self.frame_index.set(self.frame_limits[0])

        # Set the frame index slider 
        self.S0.configure(from_=self.frame_limits[0], 
            to=self.frame_limits[1])

        # Get all localizations in this interval
        self.locs = self.all_locs[self.all_locs['frame_idx'].isin(
            np.arange(self.frame_limits[0], self.frame_limits[1]))]

        # Get the positions of these localizations
        self.positions = np.asarray(self.locs[['frame_idx', 'y0', 'x0']])

        # Set the current spot filter
        self.condition = self.all_condition[self.all_locs['frame_idx'].isin(
            np.arange(self.frame_limits[0], self.frame_limits[1]))]

    def _set_vmin_vmax(self, vmin=None, vmax=None):
        """
        Change vmin or vmax, the contrast settings
        for the current image. 

        """
        # Save these values 
        if not vmin is None:
            self.vmin = float(vmin)
        if not vmax is None:
            self.vmax = float(vmax) 

        # Reload the current frame
        self._set_frame(self.frame_index.get())

    def _set_frame(self, frame_index=None):
        """
        Change the current frame. This does the following:

            1. If frame index is not passed, get the
                current value of self.frame_index.
            2. Get the raw image from the reader.
            3. Overlay spots, if self.overlay_locs is 
                set.
            4. Show the result on the canvas.

        """
        # If not given a frame index, default to the 
        # current frame
        if frame_index is None:
            frame_index = self.frame_index.get()
        else:
            self.frame_index.set(int(frame_index))

        # Get this frame from the underlying file
        # reader and scale by self.vmin, self.vmax
        self.raw_img = self._get_scaled_frame(frame_index)

        # Overlay locs if desired
        if self.overlay_locs:
            self.img = self._overlay()
        else:
            self.img = upsample(self.raw_img,
                upsampling=self.upsampling)

        # Make a photo object
        self.photo = self._get_photo(self.img)

        # Put this photo on the canvas
        self.canvas.create_image(0, 0, image=self.photo,
            anchor=tkinter.NW)

    def _overlay(self):
        """
        Given the current frame (stored at self.frame_index)
        and the current localizations (stored at self.locs)
        and the current filtering condition (stored at 
        self.condition), overlay localizations onto the 
        raw image.

        Localizations that pass the filter are colored in 
        white, while localizations that do not pass are 
        colored in red.

        """
        # Get frame index
        i = self.frame_index.get()

        # Get the current raw 8-bit frame
        raw_img = self.raw_img 

        # If there are no localizations in the present
        # frame range, return the image unchanged
        if len(self.positions.shape)<2 or self.positions.shape[0]==0:
            return raw_img 

        # Get the localizations in this frame
        in_frame = self.positions[:,0] == i 
        pos_in_frame = self.positions[in_frame, 1:]

        # If there are no localizations in the present
        # frame, return the image unchanged
        if len(pos_in_frame.shape) < 2 or pos_in_frame.shape[0] == 0:
            return raw_img 

        # Get the YX positions of each localization in the
        # two categories - passed filter or not passed 
        pos_0 = pos_in_frame[self.condition[in_frame], :]
        pos_1 = pos_in_frame[~self.condition[in_frame], :]

        # Overlay the localizations
        img = upsample_overlay_two_color(raw_img, pos_0,
            pos_1, upsampling=self.upsampling, crosshair_len=8)

        return img 

    def _get_scaled_frame(self, frame_index):
        """
        Get a frame from the underlying movie,
        and apply the current vmax/vmin filters.

        args
        ----
            frame_index : int, the frame index

        returns
        -------
            3D ndarray of shape (img.shape[0], 
                img.shape[1], 4), dtype uint8

        """
        # If no reader specified, return all zeros
        if not isinstance(self.reader, qio.ImageFileReader):
            return np.zeros((128, 128, 4), dtype='uint8')

        # Get this frame
        img = self.reader.get_frame(frame_index).astype('float64')

        # 32-bit RGBA result
        result = np.zeros((img.shape[0], img.shape[1],
            4), dtype='uint8')

        # Scale by the current vmin and vmax
        if (self.vmin == self.vmax):
            result[:,:,3] = np.full(img.shape, 255, dtype='uint8')
        else:
            emax = self.vmax - self.vmin 
            img = 255.0 * (img-self.vmin)/emax 

            # Set negative values to zero 
            img[img < 0] = 0

            # Set values over vmax to vmax 
            img[img>255.0] = 255.0

            # Convert to uint8
            result[:,:,3] = img.astype('uint8')

        # RGBA: reverse hues for imshow 
        result[:,:,3] = 255 - result[:,:,3]

        return result 

    def _get_photo(self, img):
        """
        Given a 2D ndarray in 32-bit RGBA, 
        make a PIL.ImageTk object that can be 
        plotted on a tkinter.Canvas.

        This resizes the image to fit in the 
        QCGUI's canvas object.

        args
        ----
            img : 3D ndarray, dtype uint8,
                shape (N, M, 4)

        returns
        -------
            PIL.ImageTk 

        """
        return PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(
                img
            ).resize((self.img_shape)[::-1]),
        )

class TrackGUI(object):
    def __init__(self, loc_file, image_file, root=None,
        subregion=None, frame_limits=[0, 100], upsample=3,
        n_colors=1028, half_pixel_shift=True, pixel_size_um=0.16,
        frame_interval_sec=0.00548):
        """
        init
        ----
            loc_file   : str, path to *locs.csv
            image_file : str, path to image file
                            (.nd2 or .tif)
            root       : tkinter parent GUI
            subregion  : [[int, int], [int, int]], the
                         subregion of the image to use 
            frame_limits : [int, int], the initial frame
                         limits
            upsample   : int, upsampling factor
            n_colors   : int, the number of colors (must match 
                         utils.n_colors)
            half_pixel_shift: bool, shift all locs by +0.5 
                         pixels
            pixel_size_um : float, pixel size in um
            frame_interval_sec : float, time between frames

        """
        ##
        ## ATTRIBUTES
        ## 
        self.static_configs = {
            'pixel_size_um': pixel_size_um,
            'frame_interval_sec': frame_interval_sec,
        }

        # Shift all positions by half a pixel.
        self.half_pixel_shift = half_pixel_shift 

        # Source files
        self.loc_file = loc_file
        self.image_file = image_file 

        # Current directory, matching the loc file
        self.curr_dir = os.path.dirname(self.loc_file)

        # Size of the GUI window 
        self.gui_width = 900
        self.gui_height = 300

        # Position of the images in the canvas subwindow
        delta_x = self.gui_width // 3
        delta_y = self.gui_height // 3
        self.image_pos = [[0, 0], [delta_x+1, 0], [2*(delta_x+1), 0]]

        # Start the main tkinter interface
        if root is None:
            root = tkinter.Tk()
        self.root = root 

        # The number of unique colors; must match utils.n_colors
        self.n_colors = n_colors 

        # Image file reader
        self.reader = qio.ImageFileReader(image_file)

        # Shape of the underlying movie
        self.n_frames, self.IH, self.IW = self.reader.get_shape()

        # If no subregion is passed, default to full frame
        if subregion is None:
            self.subregion = [[0, self.IH], [0, self.IW]]
        else:
            self.subregion = subregion 

        # Shape of the subregion
        self.SH = self.subregion[0][1] - self.subregion[0][0]
        self.SW = self.subregion[1][1] - self.subregion[1][0]

        # If no frame limits are passed, default to the first
        # 100 frames
        if frame_limits is None:
            self.frame_limits = [0, self.n_frames]
        else:
            self.frame_limits = frame_limits
            self.frame_limits[1] = min([
                self.frame_limits[1], self.n_frames])

        # Resizing factor for images, so that they fill
        # image width
        self.resize_factor = min([
            300/self.SW,
            300/self.SH 
        ])

        # Image shape, for passing to PIL.Image.resize
        self.region_shape = np.array([
            self.resize_factor * self.SH,
            self.resize_factor * self.SW, 
        ]).astype('uint16')

        # Upsampling factor
        self.upsample = upsample 

        # Current frame index
        self.frame_index = tkinter.IntVar(self.root)
        self.frame_index.set(self.frame_limits[0])

        # Raw localization dataframe
        self.locs = pd.read_csv(self.loc_file)

        # Assign each localization a unique frame
        # index, to save us doing this at each
        # tracking iteration
        self.locs.loc[:,'loc_idx'] = np.arange(len(self.locs))

        # By default, give all localizations the 
        # same trajectory index
        self.locs.loc[:,'traj_idx'] = np.ones(len(self.locs))

        # Record all of the boolean columns
        self.bool_cols = [c for c in self.locs.columns
            if self.locs[c].dtype == 'bool']

        # Options for menu M1 
        self.filter_options = ["None"] + self.bool_cols

        # Set self.sublocs, the part of the dataframe
        # that coincides with the current frame
        # range. This also sets self.positions, which 
        # is an ndarray with the raw positions and frames
        # for fast lookup. It has the following column
        # organization:
        #    self.positions[:,0] -> frame index 
        #    self.positions[:,1] -> y position (pixels)
        #    self.positions[:,2] -> x position (pixels)
        #    self.positions[:,3] -> trajectory index 
        #    self.positions[:,4] -> color index 
        self._set_sublocs()

        # Current vmax
        self.vmax = tkinter.DoubleVar(self.root)

        # Current vmin
        self.vmin = tkinter.DoubleVar(self.root)

        # Set vmax and vmin according to the maximum and
        # minimum for the selected frame range
        mm, mx = self.reader.min_max(start_frame=self.frame_limits[0],
            stop_frame=self.frame_limits[1])
        self.vmax.set(mx)
        self.vmin.set(mm)

        # Current algorithm type
        self.algorithm_type = tkinter.StringVar(self.root)
        self.algorithm_type.set("diffusion_only")

        # Available algorithm types
        self.algorithm_types = DEFAULT_TRACK_KWARGS.keys()

        # Current filtering option
        self.filter_option = tkinter.StringVar(self.root)
        self.filter_option.set("None")

        # The current type of crosshairs
        self.crosshair_type = tkinter.StringVar(self.root)
        self.crosshair_type.set('+')

        # The available types of crosshairs
        self.crosshair_choices = ['+', 'o', '+ with history',
            'o with history']

        # Whether we are currently overlaying trajectories
        self.overlay_trajs = False 

        # Set vmax/vmin to the max and min in the
        # frame range
        mmin, mmax = self.reader.min_max(
            start_frame=self.frame_limits[0],
            stop_frame=self.frame_limits[1],
        )
        self.vmax.set(mmax)
        self.vmin.set(mmin)


        ##
        ## TKINTER FORMAT KWARGS
        ##
        slider_kwargs = {'orient': tkinter.HORIZONTAL,
            'activebackground': '#000000', 'resolution': 1,
            'length': 200, 'font': ('Helvetica', '16'),
            'sliderrelief': tkinter.FLAT, 'bd': 0}
        label_kwargs = {'font': ('Helvetica', 16)}
        button_kwargs = {'activeforeground': '#dadde5',
            'activebackground': '#455462'}
        grid_kwargs = {'sticky': tkinter.NW, 'pady': 10,
            'padx': 10}


        ##
        ## LAYOUT
        ##

        # Master frame: for organizing the other frames
        self.master_frame = tkinter.Frame(self.root, 
            height=2*self.gui_height, width=self.gui_width)
        self.master_frame.pack()

        # Upper frame, for images 
        self.upper_frame = tkinter.Frame(self.master_frame,
            height=self.gui_height, width=self.gui_width)
        self.upper_frame.pack()

        # Lower frame, for buttons and stuff
        self.lower_frame = tkinter.Frame(self.master_frame,
            height=self.gui_height, width=self.gui_width)
        self.lower_frame.pack()

        # Canvas, raw canvas for images in upper frame
        self.canvas = tkinter.Canvas(self.upper_frame, 
            height=self.gui_height, width=self.gui_width)
        self.canvas.grid(row=0, column=0)


        ## LEFT COLUMN OF BUTTONS

        # Slider S0, which controls frame index
        self.S0 = tkinter.Scale(self.lower_frame, 
            from_=self.frame_limits[0], to=self.frame_limits[1]-2,
            command=self._S0_callback, label='Frame index',
            **slider_kwargs)
        self.S0.grid(row=0, column=0, **grid_kwargs)
        self.S0.set(self.frame_index.get())

        # Slider S1, which controls vmax 
        self.S1 = tkinter.Scale(self.lower_frame, 
            from_=0.0, to=self.vmax.get(),
            command=self._S1_callback, label='vmax',
            **slider_kwargs)
        self.S1.grid(row=1, column=0, **grid_kwargs)
        self.S1.set(self.vmax.get())

        # Slider S2, which controls vmin 
        self.S2 = tkinter.Scale(self.lower_frame, 
            from_=0.0, to=self.vmax.get(),
            command=self._S2_callback, label='vmin',
            **slider_kwargs)
        self.S2.grid(row=2, column=0, **grid_kwargs)
        self.S2.set(self.vmin.get())

        # Label LM0, which is the label for menu M0
        self.LM0 = tkinter.Label(self.lower_frame,
            text='Algorithm type').grid(row=3, column=0,
            sticky=tkinter.NW, pady=5, padx=5)

        # Menu M0, which selects the tracking option
        self.M0 = tkinter.OptionMenu(self.lower_frame,
            self.algorithm_type, *self.algorithm_types,
            command=self._M0_callback)
        self.M0.grid(row=4, column=0, **grid_kwargs)

        # Label LM1, which is the label for menu M1 
        self.LM1 = tkinter.Label(self.lower_frame,
            text='Apply filter').grid(row=5, column=0,
            sticky=tkinter.NW, pady=5, padx=5)

        # Menu M1, which selects the filter option
        self.M1 = tkinter.OptionMenu(self.lower_frame,
            self.filter_option, *self.filter_options)
        self.M1.grid(row=6, column=0, **grid_kwargs)


        ## MIDDLE COLUMN OF BUTTONS

        # Sliders S3 through S6, which are semantically
        # flexible and respond to settings on the current
        # tracking algorithm
        self.S3 = tkinter.Scale(self.lower_frame, 
            from_=0, to=10, command=self._S3_callback,
            **slider_kwargs)
        self.S3.grid(row=0, column=1, **grid_kwargs)

        self.S4 = tkinter.Scale(self.lower_frame, 
            from_=0, to=10, command=self._S4_callback,
            **slider_kwargs)
        self.S4.grid(row=1, column=1, **grid_kwargs)       
  
        self.S5 = tkinter.Scale(self.lower_frame, 
            from_=0, to=10, command=self._S5_callback,
            **slider_kwargs)
        self.S5.grid(row=2, column=1, **grid_kwargs)

        self.S6 = tkinter.Scale(self.lower_frame, 
            from_=0, to=10, command=self._S6_callback,
            **slider_kwargs)
        self.S6.grid(row=3, column=1, **grid_kwargs)


        ## RIGHT COLUMN OF BUTTONS

        # Button B0, which toggles the overlay
        self.B0 = tkinter.Button(self.lower_frame,
            text='Overlay trajs', command=self._B0_callback,
            **button_kwargs)
        self.B0.grid(row=0, column=2, **grid_kwargs)

        # Label LM2, which is the label for menu M2
        self.LM2 = tkinter.Label(self.lower_frame,
            text='Tracking crosshair type').grid(
            row=1, column=2, sticky=tkinter.NW,
            pady=5, padx=5)

        # Menu M2, which changes the type of 
        # tracking crosshair
        self.M2 = tkinter.OptionMenu(self.lower_frame,
            self.crosshair_type, *self.crosshair_choices,
            command=self._M2_callback)
        self.M2.grid(row=2, column=2, **grid_kwargs)

        # Button B1, which saves the current tracking
        # settings
        self.B1 = tkinter.Button(self.lower_frame,
            text='Save current settings', command=self._B1_callback,
            **button_kwargs)
        self.B1.grid(row=3, column=2, **grid_kwargs)

        # Button B2, which changes the frame range
        self.B2 = tkinter.Button(self.lower_frame,
            text='Change frame range', command=self._B2_callback,
            **button_kwargs)
        self.B2.grid(row=4, column=2, **grid_kwargs)


        ##
        ## INSTANTIATION
        ##

        # Update the frame to the initial image
        self._update_frame()

        # Run the main loop
        self.root.mainloop()

    def _S0_callback(self, frame_index):
        """
        Respond to changes in slider S0, 
        which controls frame index.

        """
        self._update_frame(frame_index=int(frame_index))

    def _S1_callback(self, vmax):
        """
        Respond to changes in slider S1,
        which controls vmax.

        """
        self._update_frame(vmax=float(vmax))

    def _S2_callback(self, vmin):
        """
        Respond to changes in slider S2, 
        which controls vmin.

        """
        self._update_frame(vmin=float(vmin))

    def _M0_callback(self, *args):
        """
        Response to changing the menu M0, which 
        sets the type of tracking algorithm.

        Update the values of self.configs, which 
        are the tracking settings, and retrack
        the localizations at self.sublocs.

        Then update the frame.

        """
        # Current algorithm type 
        v = self.algorithm_type.get()

        # Set default algorithm settings
        self.configs = DEFAULT_TRACK_KWARGS[v]

        # Add static attributes like frame interval
        # and pixel size
        self.configs = {**self.configs, **self.static_configs}

        # Change the identities of the sliders 
        slider_ids = TRACK_SLIDER_IDENTITIES[v]
        n = len(slider_ids)

        slider_ref = [self.S3, self.S4, self.S5, self.S6]
        for i in range(n):
            slider_ref[i].configure(
                from_=TRACK_SLIDER_LIMITS[v][i][0],
                to=TRACK_SLIDER_LIMITS[v][i][1],
                resolution=TRACK_SLIDER_RESOLUTIONS[v][i],
                label=slider_ids[i],
            )
            slider_ref[i].set(self.configs[slider_ids[i]])

        # Retrack the localizations
        self._track()

        # Update the current frame
        self._update_frame()

    def _S3_callback(self, value):
        """
        Respond to changing slider S3, which is configured
        by the current algorithm type.

        """
        slider_ids = TRACK_SLIDER_IDENTITIES[self.algorithm_type.get()]
        n = len(slider_ids)
        if n < 1:
            return 
        else:
            self.configs[slider_ids[0]] = float(value)
            self._track()
            self._update_frame()

    def _S4_callback(self, value):
        """
        Responds to changing slider S4, which is configured
        by the current algorithm type.

        """
        slider_ids = TRACK_SLIDER_IDENTITIES[self.algorithm_type.get()]
        n = len(slider_ids)
        if n < 2:
            return 
        else:
            self.configs[slider_ids[1]] = float(value)
            self._track()
            self._update_frame()

    def _S5_callback(self, value):
        """
        Responds to changing slider S5, which is configured
        by the current algorithm type.

        """
        slider_ids = TRACK_SLIDER_IDENTITIES[self.algorithm_type.get()]
        n = len(slider_ids)
        if n < 3:
            return 
        else:
            self.configs[slider_ids[2]] = float(value)
            self._track()
            self._update_frame()

    def _S6_callback(self, value):
        """
        Responds to changing slider S5, which is configured
        by the current algorithm type.

        """
        slider_ids = TRACK_SLIDER_IDENTITIES[self.algorithm_type.get()]
        n = len(slider_ids)
        if n < 4:
            return 
        else:
            self.configs[slider_ids[3]] = float(value)
            self._track()
            self._update_frame()

    def _B0_callback(self, *args):
        """
        Respond to pressing button B0, which 
        triggers trajectory overlay.

        """
        self.overlay_trajs = not self.overlay_trajs
        self._update_frame()

    def _M2_callback(self, *args):
        """
        Respond to changes in menu M2, which 
        sets the type of tracking crosshair.

        """
        # Update the frame
        self._update_frame()

    def _B1_callback(self, *args):
        """
        Respond to pressing button B1, which 
        saves the current tracking settings.

        """
        # Format current configs for saving
        track_configs = {'tracking': self.configs}

        # Prompt the user to select either an 
        # existing file or a new file
        options = ["Append to existing settings file",
            "Write new settings file"]
        choice = OptionPrompt(options, title='Save type')

        if choice == 'Append to existing settings file':

            # Prompt the user to select a settings file
            config_file = filedialog.askopenfilename(
                initialdir=self.curr_dir,
                title='Select settings file to append',
                filetypes=(("yaml files", "*.yaml"),
                    ("all files", "*.*"))
            )

            # Open this config file
            configs = qio.read_config(config_file)

            # Append to the existing config file
            configs['tracking'] = track_configs['tracking']

            # Save to the same file
            qio.save_config(config_file, configs)

        elif choice == 'Write new settings file':

            # Prompt the user to write a filename
            save_file = filedialog.asksaveasfilename(
                parent=self.root,
                initialdir=self.curr_dir,
                defaultextension='.yaml'
            )

            # Write to the file
            qio.save_config(save_file, track_configs)

    def _B2_callback(self, *args):
        """
        Respond to pressing button B2, which changes
        the frame range.

        """
        # Prompt the user to select 
        defaults = {'Start frame': self.frame_limits[0],
            'Stop frame': self.frame_limits[1]}
        choices = VariableEntryPrompt(defaults, 
            title='Select frame range')

        # Change the frame limits
        self.frame_limits[0] = choices['Start frame']
        self.frame_limits[1] = choices['Stop frame']

        # Set the current frame index to the start
        self.frame_index.set(self.frame_limits[0])

        # Configure the frame index slider
        self.S0.configure(from_=self.frame_limits[0],
            to=self.frame_limits[1]-2)

        # Reset the current sub localization dataframe
        self._set_sublocs()

        # Re-track
        self._track()

        # Update the current frame
        self._update_frame()

    def _track(self):
        """
        Using the current configuration settings at 
        self.configs, run tracking on the localizations
        at self.sublocs.

        """
        # Track only localizations in this frame range
        self.sublocs = track.track_locs(self.sublocs, 
            **self.configs)

        # Rewrite the self.positions attribute
        self.positions = np.asarray(self.sublocs[['frame_idx',
            'y0', 'x0', 'traj_idx', 'loc_idx']])
        self.positions[:,4] = (self.positions[:,3]*717) \
            % self.n_colors

        # Shift by half a pixel, if desired
        if self.half_pixel_shift:
            self.positions[:,1:3] = self.positions[:,1:3]+0.5

    def _set_sublocs(self):
        """
        Set self.sublocs to localizations from the current
        frame range, stored at self.frame_limits.

        """
        # Dataframe
        self.sublocs = self.locs[self.locs['frame_idx'].isin(
            np.arange(self.frame_limits[0], self.frame_limits[1]))]
        self.sublocs.loc[:, 'loc_idx'] = np.arange(len(self.sublocs))

        # Ndarray, for fast position lookup
        self.positions = np.asarray(self.sublocs[['frame_idx',
            'y0', 'x0', 'traj_idx', 'loc_idx']])
        self.positions[:,4] = (self.positions[:,3]*717) \
            % self.n_colors

        # Shift by half a pixel, if desired
        if self.half_pixel_shift:
            self.positions[:,1:3] = self.positions[:,1:3]+0.5

    def _set_frame_index(self, frame_index):
        """
        Change the current frame index.

        """
        self._update_frame(frame_index=int(frame_index))

    def _set_vmax(self, vmax):
        """
        Change the current vmax.

        """
        self._update_frame(vmax=float(vmax))

    def _set_vmin(self, vmin):
        """
        Change the current vmin.

        """
        self._update_frame(vmin=float(vmin))

    def _update_frame(self, frame_index=None, vmax=None,
        vmin=None):
        """
        Main function to update the current frame.

        Sequentially:

            1. Grab the frame from the reader. If
                frame_idx is None, use self.frame_index.

            2. If self.overlay_trajs is True, get the 
                localizations corresponding to this frame
                from self.positions and their respective
                trajectory indices from self.traj_indices.

            3. Rescale the image according to self.vmax
                and self.vmin (done by utils.upsample_overlay_trajs).

            4. Convert to 8-bit (done by 
                utils.upsample_overlay_trajs).

            5. Upsample according to self.upsample (done by 
                utils.upsample_overlay_trajs).

            6. If self.overlay_trajs is True, overlay
                the localizations onto the image with the 
                different colors for different trajectories.

            7. Convert to PIL.ImageTk.PhotoImage and assign
                to the respective self.canvas locations.

        """
        # Get the frame index
        if frame_index is None:
            frame_index = self.frame_index.get()
        else:
            self.frame_index.set(frame_index)

        # Get vmax
        if vmax is None:
            vmax = self.vmax.get()
        else:
            self.vmax.set(vmax)

        # Get vmin
        if vmin is None:
            vmin = self.vmin.get()
        else:
            self.vmin.set(vmin)

        # Get the current crosshair type
        ct = self.crosshair_type.get()

        # Get this frame and the next two (WORKING)
        self.img_0 = self.reader.get_frame(frame_index)
        self.img_1 = self.reader.get_frame(frame_index+1)
        self.img_2 = self.reader.get_frame(frame_index+2)

        if self.overlay_trajs:

            # Determine which localizations correspond
            # to these three frames
            in0 = self.positions[:,0]==frame_index
            in1 = self.positions[:,0]==(frame_index+1)
            in2 = self.positions[:,0]==(frame_index+2)

            # Get the corresponding positions 
            pos0 = self.positions[in0,1:3]
            pos1 = self.positions[in1,1:3]
            pos2 = self.positions[in2,1:3]

            # Get the corresponding traj indices, 
            # which determine the traj LUT
            colors0 = self.positions[in0,4].astype('int64')
            colors1 = self.positions[in1,4].astype('int64')
            colors2 = self.positions[in2,4].astype('int64')

            # Options that do not overlay the full trajectory
            if ct in ['+', 'o']:

                # Make the PIL.ImageTk.PhotoImage objects 
                self.photo_0 = self._get_photo(utils.upsample_overlay_trajs(
                    self.img_0, pos=pos0, traj_indices=colors0, vmax=vmax,
                    vmin=vmin, u=self.upsample, crosshair_type=ct))
                self.photo_1 = self._get_photo(utils.upsample_overlay_trajs(
                    self.img_1, pos=pos1, traj_indices=colors1, vmax=vmax,
                    vmin=vmin, u=self.upsample, crosshair_type=ct))           
                self.photo_2 = self._get_photo(utils.upsample_overlay_trajs(
                    self.img_2, pos=pos2, traj_indices=colors2, vmax=vmax,
                    vmin=vmin, u=self.upsample, crosshair_type=ct))  

            # Options that must incorporate the past history 
            # of each trajectory
            elif ct in ['+ with history', 'o with history']:

                # For each trajectory, get its past history
                histories0 = [self.positions[np.logical_and(
                    self.positions[:,4]==i, self.positions[:,0]<=frame_index
                ),1:3] for i in colors0]
                histories1 = [self.positions[np.logical_and(
                    self.positions[:,4]==i, self.positions[:,0]<=frame_index+1
                ),1:3] for i in colors1]
                histories2 = [self.positions[np.logical_and(
                    self.positions[:,4]==i, self.positions[:,0]<=frame_index+2
                ),1:3] for i in colors2]

                # Make the PIL.ImageTk.PhotoImage objects 
                self.photo_0 = self._get_photo(utils.upsample_overlay_trajs_history(
                    self.img_0, pos_history=histories0, traj_indices=colors0, vmax=vmax,
                    vmin=vmin, u=self.upsample, crosshair_type=ct[0]))
                self.photo_1 = self._get_photo(utils.upsample_overlay_trajs_history(
                    self.img_1, pos_history=histories1, traj_indices=colors1, vmax=vmax,
                    vmin=vmin, u=self.upsample, crosshair_type=ct[0]))           
                self.photo_2 = self._get_photo(utils.upsample_overlay_trajs_history(
                    self.img_2, pos_history=histories2, traj_indices=colors2, vmax=vmax,
                    vmin=vmin, u=self.upsample, crosshair_type=ct[0]))

        else:
            self.photo_0 = self._get_photo(utils.upsample_overlay_trajs(
                self.img_0, vmax=vmax, vmin=vmin, u=self.upsample,
                crosshair_type=ct))
            self.photo_1 = self._get_photo(utils.upsample_overlay_trajs(
                self.img_1, vmax=vmax, vmin=vmin, u=self.upsample,
                crosshair_type=ct))
            self.photo_2 = self._get_photo(utils.upsample_overlay_trajs(
                self.img_2, vmax=vmax, vmin=vmin, u=self.upsample,
                crosshair_type=ct))

        # Assign the photos to their respective
        # canvas positions
        self.canvas.create_image(
            self.image_pos[0][0],
            self.image_pos[0][1],
            image=self.photo_0,
            anchor=tkinter.NW)
        self.canvas.create_image(
            self.image_pos[1][0],
            self.image_pos[1][1],
            image=self.photo_1,
            anchor=tkinter.NW)
        self.canvas.create_image(
            self.image_pos[2][0],
            self.image_pos[2][1],
            image=self.photo_2,
            anchor=tkinter.NW)

    def _get_photo(self, img):
        """
        Convert an image from 8-bit RGBA to 
        PIL.ImageTk.PhotoImage.

        """
        # Make a PhotoImage
        return PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(
                img,
            ).resize(self.region_shape),
        )
        
def get_photo(img, expand=1, vmax=255.0, resize=1.0):
    if resize is None:
        return PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(
                img_to_rgb(img, expand=expand, vmax=vmax),
            ),
        ) 
    else:
        shape = np.asarray(img.shape)
        return PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(
                img_to_rgb(img, expand=expand, vmax=vmax),
            ).resize((shape*resize).astype('uint16')[::-1]),
        ) 

def VariableEntryPrompt(defaults, title=None):
    """
    Launch a mini GUI that prompts the user to 
    enter text in boxes that are labeled with
    different names. 

    args
    ----
        default : dict of str : value, the
            names of each box mapped to their
            default value. The type of the default
            value also determines the return type.

        title : str 

    returns
    -------
        dict of str : value, the user-modified
            values

    """
    # Format kwargs
    button_kwargs = {'activeforeground': '#dadde5',
        'activebackground': '#455462'}
    grid_kwargs = {'pady': 5, 'padx': 5,
        'sticky': tkinter.NW}

    # Make the main interface
    top = tkinter.Toplevel()
    if not title is None:
        top.title(title)
    frame = tkinter.Frame(top, height=300, width=200)
    frame.pack()

    # Get the names of the boxes
    keys = [str(i) for i in defaults.keys()]
    n = len(keys)

    # Get the initial value for each box
    initials = [defaults[k] for k in keys]

    # Get the return types
    types = [type(i) for i in initials]

    # Make the return variables
    R = [tkinter.StringVar() for i in range(n)]

    # Make a set of labels
    labels = []
    for i in range(n):
        L = tkinter.Label(frame, text=keys[i])
        L.grid(row=i, column=0, **grid_kwargs)
        labels.append(L)

    # Make a set of entry boxes
    for i in range(n):
        E = tkinter.Entry(frame, textvariable=R[i])
        E.grid(row=i, column=1)

        # Set the default value
        E.insert(tkinter.END, str(initials[i]))

    # Make a destroy button
    def _close(): top.destroy()
    B = tkinter.Button(frame, text='Accept',
        command=_close)
    B.grid(row=n, column=1)

    # Run the GUI
    top.wait_window()

    # Format the outputs 
    values = [convert_on_example(r.get(), i) for r, i \
        in zip(R, initials)]

    return {k: v for k, v in zip(keys, values)}

def OptionPrompt(options, title=None, label=None):
    """
    Prompt the user to select one of several options
    from a drop-down menu, returning the chosen option
    as a string.

    """
    # Window
    top = tkinter.Toplevel()
    if not title is None:
        top.title(title)

    # Result
    result = tkinter.StringVar()
    result.set(options[0])

    # Label
    if not label is None:
        L = tkinter.Label(top, text=label)
        L.pack(pady=20, padx=20)

    # Option menu 
    M = tkinter.OptionMenu(top, result, 
        *options)
    M.pack(pady=20, padx=20)

    # Kill button
    def _close(): top.destroy()
    B = tkinter.Button(top, text='Accept',
        command=_close)
    B.pack(pady=20, padx=20)

    # Run GUI
    top.wait_window()

    # Format output
    return result.get()

def OptionsPrompt(option_lists, labels, title=None):
    """
    Prompt the user to select something from each 
    of multiple lists.

    """
    n = len(option_lists)
    assert n == len(labels)

    # Window
    top = tkinter.Toplevel()
    if not title is None:
        top.title(title)

    # Frame
    frame = tkinter.Frame(top, width=200, height=400)
    frame.pack()

    # Results
    results = [tkinter.StringVar() for i in range(n)]
    for j, result in enumerate(results):
        results[j].set(option_lists[j][0])

    # Label
    label_objs = []
    for j, label in enumerate(labels):
        L = tkinter.Label(frame, text=labels[j])
        L.grid(row=j, column=0, pady=10, padx=10)
        label_objs.append(L)

    # Option menus
    menu_objs = []
    for j in range(len(option_lists)):
        menu_objs.append(tkinter.OptionMenu(frame, results[j], 
            *option_lists[j]).grid(row=j, column=1,
            pady=10, padx=10))

    # Kill button
    def _close(): top.destroy()
    B = tkinter.Button(frame, text='Accept',
        command=_close).grid(row=n, column=1, pady=10,
        padx=10)

    # Run GUI
    top.wait_window()

    # Format output
    return [result.get() for result in results]

class DrawGUI(object):
    """
    Launch a GUI to draw on a raw image.

    The result is saved in the .path attribte of the 
    GUI.

    init
    ----
        img : 2D ndarray
        vmax_mod : float

    """
    def __init__(self, img, vmax_mod=1.0, gui_size=500,
        axis_labels_yx=None):
        self.img = img 

        # Format size 
        self.shape = np.array(self.img.shape)
        self.gui_size = gui_size 
        self.resize_factor = min(self.gui_size/self.shape)
        self.shape = (self.shape * self.resize_factor).astype('int64')

        # State of the clicker
        self.b1 = 'up'

        # Last clicks
        self.xold = None 
        self.yold = None 

        # Rescale to 8-bit
        self.img = self.img * 255.0 / (self.img.max()*vmax_mod)
        self.img[self.img > 255.0] = 255.0
        self.img = self.img.astype('uint8')

        # Make drawing window
        self.root = tkinter.Toplevel()
        self.canvas = tkinter.Canvas(self.root, 
            width=self.gui_size, height=self.gui_size)
        self.canvas.pack()

        # Make photo
        self.photo = self.get_photo(self.img)
        self.canvas.create_image(0, 0, image=self.photo,
            anchor=tkinter.NW)

        # Label with info for user 
        if not axis_labels_yx is None:
            self.L = tkinter.Label(self.root, text=
                "y axis: %s, x_axis: %s" % tuple(axis_labels_yx))
            self.L.pack()

        # Drawn path 
        self.path = []

        # Bindings to mouse
        self.canvas.bind("<Motion>", self.motion)
        self.canvas.bind("<ButtonPress-1>", self.b1down)
        self.canvas.bind("<ButtonRelease-1>", self.b1up)

        self.root.wait_window()

        self.path = np.asarray(self.path) / self.resize_factor 

    def b1down(self, event):
        self.b1 = 'down'
        self.path = []
        self.canvas.delete('line')

    def b1up(self, event):
        self.b1 = 'up'
        self.xold = None
        self.yold = None 

    def motion(self, event):
        if self.b1 == 'down':
            if self.xold is not None and self.yold is not None:
                event.widget.create_line(self.xold, self.yold,
                    event.x, event.y, smooth=tkinter.TRUE,
                    fill='white', tag='line')
            self.xold = event.x 
            self.yold = event.y 
            self.path.append((self.xold, self.yold))

    def get_photo(self, img):
        return PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(img).resize(self.shape)
        )

def convert_on_example(str_arg, example):
    """
    Try to convert *str_arg* to the same type
    as *example*.

    args
    ----
        str_arg : str
        example : either int, float, bool, or
                    str

    returns
    -------
        type matching example if success, or 
            str if fail

    """
    if isinstance(example, int):
        try:
            return int(str_arg)
        except ValueError:
            return str_arg 
    elif isinstance(example, float):
        try:
            return float(str_arg)
        except ValueError:
            return str_arg 
    elif isinstance(example, bool):
        if (str_arg=='True') or (str_arg=='true'):
            return True 
        elif (str_arg=='False') or (str_arg=='false'):
            return False 
        else:
            return str_arg 
    else:
        return str_arg 


def img_to_rgb(img, expand=1, vmax=255.0, vmin=0.0):
    """
    Convert a 2D ndarray to 32-bit RGBA, 
    which is expected by PIL.Image.fromarray.

    args
    ----
        img :  2D ndarray
        expand :  int, the expansion factor

    returns
    -------
        3D ndarray of shape (img.shape[0], 
            img.shape[1], 4)

    """
    # Adjust intensities to fit in 0-255 range
    img_re = (img.astype('float64')*255.0/ \
        max([img.max(),1]))
    img_re = set_neg_to_zero(img_re-vmin)

    # If vmax is below the minimum intensity,
    # return all 255. Else rescale.
    if vmax < img_re.min():
        img_re = np.full(img.shape, 255, dtype='uint8')
    else:
        img_re = img_re * 255.0 / vmax 
        img_re[img_re > 255.0] = 255.0
        img_re = img_re.astype('uint8')
    N, M = img_re.shape 

    # Make RGB rep
    img_rgb = np.zeros((N, M, 4), dtype='uint8')
    img_rgb[:,:,3] = 255 - img_re 

    # Expand, if desired
    if expand>1:
        exp_img_rgb = np.zeros((N*expand, M*expand, 4), dtype='uint8')
        for i in range(expand):
            for j in range(expand):
                exp_img_rgb[i::expand, j::expand] = img_rgb 
        return exp_img_rgb 
    else:
        return img_rgb 

def get_photo(img, expand=1, vmax=255.0, resize=1.0):
    if resize is None:
        return PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(
                img_to_rgb(img, expand=expand, vmax=vmax),
            ),
        ) 
    else:
        shape = np.asarray(img.shape)
        return PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(
                img_to_rgb(img, expand=expand, vmax=vmax),
            ).resize((shape*resize).astype('uint16')[::-1]),
        ) 

def find_and_overlay_spots(img, img_bin, crosshair_len=4):
    """
    Find contiguous binary spots in a binary image
    and overlay them on a different image. Returns
    a copy of the image.

    args
    ----
        img : 2D ndarray, the image on which to overlay
            locs
        img_bin : 2D ndarray, binary image in which to 
            look for spots
        crosshair_len : int, crosshair length

    returns
    -------
        2D ndarray, copy of *img* with overlaid spots

    """
    # If there are too many spots, give up - this
    # happens sometimes when the threshold is all wrong
    img_size = np.asarray(img_bin.shape).prod()
    if img_bin.sum() > img_size * 0.2:
        return img 

    # Otherwise overlay the spots
    positions = label_binary_spots(img_bin,
        img_int=img)
    result = overlay_spots(img, positions, 
        crosshair_len=crosshair_len)
    return result 

##
## GUI CONFIGS: FILTERING
##

# Default parameter sets when changing to a different
# filtering method
FILTER_KWARGS_DEFAULT = {
    'unfiltered': {},
    'sub_median': {'scale': 1.0},
    'sub_min': {'scale': 1.0},
    'sub_mean': {'scale': 1.0},
    'sub_gauss_filt_median': {'k': 2.0, 'scale': 1.0},
    'sub_gauss_filt_min': {'k': 2.0, 'scale': 1.0},
    'sub_gauss_filt_mean': {'k': 2.0, 'scale': 1.0},
}

# The mapping of filtering keyword arguments to 
# sliders S20 and S30
FILTER_KWARG_MAP = {
    'unfiltered': [],
    'sub_median': ['scale'],
    'sub_min': ['scale'],
    'sub_mean': ['scale'],
    'sub_gauss_filt_median': ['k', 'scale'],
    'sub_gauss_filt_min': ['k', 'scale'],
    'sub_gauss_filt_mean': ['k', 'scale'],
}

# Step size of the filtering sliders S20 and S30 
FILTER_SLIDER_RESOLUTIONS = {
    'unfiltered': [],
    'sub_median': [0.01],
    'sub_min': [0.01],
    'sub_mean': [0.01],
    'sub_gauss_filt_median': [0.1, 0.01],
    'sub_gauss_filt_min': [0.1, 0.01],
    'sub_gauss_filt_mean': [0.1, 0.01],
}

# Lower and upper limits for the filtering sliders
# S20 and S30 
FILTER_SLIDER_LIMITS = {
    'unfiltered': [[]],
    'sub_median': [[0.5, 1.5]],
    'sub_min': [[0.5, 1.5]],
    'sub_mean': [[0.5, 1.5]],
    'sub_gauss_filt_median': [[0.0, 10.0], [0.5, 1.5]],
    'sub_gauss_filt_min': [[0.0, 10.0], [0.5, 1.5]],
    'sub_gauss_filt_mean': [[0.0, 10.0], [0.5, 1.5]],
}

##
## GUI CONFIGS: DETECTION
##

# The detection functions, keyed by option in 
# menu M1 
DETECT_METHODS = {
    'DoG': detect.dog_filter,
    'DoU': detect.dou_filter,
    'simple_gauss': detect.gauss_filter,
    'simple_gauss_squared': detect.gauss_filter_sq,
    'min/max': detect.min_max_filter,
    'LLR': detect.llr,
    'simple_threshold': detect.simple_threshold,
}

# Default arguments to each detection function
DETECT_KWARGS_DEFAULT = {
    'DoG': {
        'k0': 1.5,
        'k1': 8.0,
        't': 400.0,
    },
    'DoU': {
        'k0': 3,
        'k1': 9,
        't': 400.0,
    },
    'simple_gauss': {
        'k': 1.0,
        't': 400.0,
    },
    'simple_gauss_squared': {
        'k': 1.0,
        't': 400.0,        
    },
    'min/max': {
        'w': 9,
        't': 400.0,
    },
    'LLR': {
        'w': 9,
        'k': 1.0,
        't': 400.0,
    },
    'simple_threshold': {
        't': 400.0,        
    },
}

# The mapping of detection keyword arguments to 
# sliders S01, S11, S21, and S31
DETECT_KWARG_MAP = {
    'DoG': ['k0', 'k1', 't'],
    'DoU': ['k0', 'k1', 't'],
    'simple_gauss': ['k', 't'],
    'simple_gauss_squared': ['k', 't'],
    'min/max': ['w', 't'],
    'LLR': ['w', 'k', 't'],
    'simple_threshold': ['t'],
}

# Step size of the detection sliders
DETECT_SLIDER_RESOLUTIONS = {
    'DoG': [0.1, 0.25, 0.1],
    'DoU': [1, 1, 0.1],
    'simple_gauss': [0.1, 0.1],
    'simple_gauss_squared': [0.1, 0.1],
    'min/max': [1, 0.1],
    'LLR': [1, 0.1, 0.1],
    'simple_threshold': [0.1],
}

# Limits on each detection slider
DETECT_SLIDER_LIMITS = {
    'DoG': [[0.0, 5.0], [0.0, 15.0], [0.0, 1000.0]],
    'DoU': [[1, 11], [1, 21], [0.0, 1000.0]],
    'simple_gauss': [[0.0, 5.0], [0.0, 1000.0]],
    'simple_gauss_squared': [[0.0, 5.0], [0.0, 1000.0]],
    'min/max': [[1, 21], [0.0, 1000.0]],
    'LLR': [[1, 21], [0.1, 3.0], [0.0, 1000.0]],
    'simple_threshold': [[0.0, 1000.0]],
}

##
## GUI CONFIGS: LOCALIZATION
##

# Localization functions
LOCALIZE_METHODS = {
    'detection_only': lambda x: x,
    'centroid': localize.centroid,
    'radial_symmetry': localize.radial_symmetry,
    'mle_poisson': localize.mle_poisson,
    'ls_int_gaussian': localize.ls_int_gaussian,
    'ls_point_gaussian': localize.ls_point_gaussian,
    'ls_log_gaussian': localize.ls_log_gaussian,
}

# Default values for localize method kwargs
LOCALIZE_KWARGS_DEFAULT = {
    'detection_only': {},
    'centroid': {
        'sub_bg': True,
        'camera_offset': 0.0,
        'camera_gain': 1.0,
    },
    'radial_symmetry': {
        'sigma': 1.0,
        'camera_offset': 0.0,
        'camera_gain': 1.0,
    },
    'mle_poisson': {
        'sigma': 1.0,
        'max_iter': 20,
        'damp': 0.3,
        'camera_offset': 0.0,
        'camera_gain': 1.0,
        'convergence_crit': 3.0e-4,
        'divergence_crit': 1.0,
        'ridge': 1.0e-4,
    },
    'ls_int_gaussian': {
        'sigma': 1.0,
        'max_iter': 20, 
        'camera_gain': 1.0,
        'camera_offset': 0.0,
        'damp': 0.3,
        'convergence_crit': 1.0e-4, 
        'divergence_crit': 1.0,
    },
    'ls_point_gaussian': {
        'sigma': 1.0,
        'max_iter': 20, 
        'camera_gain': 1.0,
        'camera_offset': 0.0,
        'damp': 0.3,
        'convergence_crit': 1.0e-4, 
        'divergence_crit': 1.0,
    },
    'ls_log_gaussian': {
        'sigma': 1.0,
        'camera_bg': 0.0,
        'camera_gain': 1.0,
    },
}

##
## GUI CONFIGS: TRACKING
##
DEFAULT_TRACK_KWARGS = {
    'conservative': {
        'algorithm_type': 'conservative',
        'd_max': 5.0,
        'min_int': 0.0,
        'search_exp_fac': 3.0,
        'max_blinks': 0,
    },
    'diffusion_only': {
        'algorithm_type': 'diffusion_only',
        'd_max': 5.0,
        'min_int': 0.0,
        'search_exp_fac': 3.0,
        'max_blinks': 0,
    },
}
TRACK_SLIDER_IDENTITIES = {
    'conservative': ['d_max', 'min_int', 'search_exp_fac', 'max_blinks'],
    'diffusion_only': ['d_max', 'min_int', 'search_exp_fac', 'max_blinks'],
}
TRACK_SLIDER_RESOLUTIONS = {
    'conservative': [0.1, 10, 0.1, 1],
    'diffusion_only': [0.1, 10, 0.1, 1],
}
TRACK_SLIDER_LIMITS = {
    'conservative': [[0.0, 20.0], [0.0, 1000.0], [0.0, 10], [0, 3]],
    'diffusion_only': [[0.0, 20.0], [0.0, 1000.0], [0.0, 10], [0, 3]],
}

