"""
gui.py

"""
# tkinter GUI
import tkinter 

# Numeric
import numpy as np 

# Plotting
import matplotlib.pyplot as plt 

# Pillow-style images, expected by tkinter
import PIL.Image, PIL.ImageTk 

# Filtering stuff
from scipy import ndimage as ndi 

# Hard copy 
from copy import copy 

# File reader 
from quot.qio import ImageFileReader 

# Image filtering/BG subtraction utilities
from quot.image_filter import SubregionFilterer

# Detection functions
from quot import detect 

# Utilities
from .utils import set_neg_to_zero, overlay_spots, label_binary_spots

sample_file_path = '/Users/alecheckert/Imaging/tracking/200321_eikon_standardize_loc_detect/nd2_files/WellB02_Channel561_Prime95B_Seq0000.nd2'
sample_subregion = [[200, 400], [200, 500]]

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
    ):
        self.filename = filename
        self.gui_height = gui_height
        self.method = method 
        self.crosshair_len = crosshair_len

        # Image file reader
        self.reader = ImageFileReader(self.filename)
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
            self.subregion, self.method)


        ## TKINTER INTERFACE COMPARTMENTALIZATION

        # Instantiate the main tkinter interface
        self.root = tkinter.Tk()
        self.root.title("quot")

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
        self.B0 = tkinter.Button(self.frame_1, text='Overlay detections',
            command=self._overlay_detections_callback)
        self.B0.grid(row=2, column=2, pady=5, padx=5)

        # The current state of this button
        self.B0_state = False 


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
            to=self.frame_limits[1], command=self._set_frame_idx,
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
    # Rescale to 8-bit
    img_re = (img.astype('float64')*255.0/ \
        max([img.max(),1]))
    img_re = set_neg_to_zero(img_re-vmin)
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


def set_neg_to_zero(ndarray):
    ndarray[ndarray < 0.0] = 0
    return ndarray 


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

# The detection functions, keyed by option in 
# menu M1 
DETECT_METHODS = {
    'DoG': detect.dog_filter,
    'DoU': detect.dou_filter,
    'simple_gauss': detect.gauss_filter,
}

# Default arguments to each detection function
DETECT_KWARGS_DEFAULT = {
    'DoG': {
        'k0': 1.5,
        'k1': 8.0,
        't': 8.0,
    },
    'DoU': {
        'k0': 3,
        'k1': 9,
        't': 8.0,
    },
    'simple_gauss': {
        'k': 1.0,
        't': 10.0,
    }
}

# The mapping of detection keyword arguments to 
# sliders S01, S11, S21, and S31
DETECT_KWARG_MAP = {
    'DoG': ['k0', 'k1', 't'],
    'DoU': ['k0', 'k1', 't'],
    'simple_gauss': ['k', 't'],
}

# Step size of the detection sliders
DETECT_SLIDER_RESOLUTIONS = {
    'DoG': [0.1, 0.25, 0.1],
    'DoU': [1, 1, 0.1],
    'simple_gauss': [0.1, 0.1],
}

# Limits on each detection slider
DETECT_SLIDER_LIMITS = {
    'DoG': [[0.0, 5.0], [0.0, 15.0], [0.0, 40.0]],
    'DoU': [[1, 11], [1, 21], [0.0, 40.0]],
    'simple_gauss': [[0.0, 5.0], [0.0, 40.0]]
}
