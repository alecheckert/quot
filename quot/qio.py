"""
qio.py

"""
import os
import tifffile
from nd2reader import ND2Reader 
from czifile import CziFile 

class ImageFileReader(object):
    def __init__(self, file_name):
        self.file_name = file_name 
        if '.nd2' in file_name:
            self.type = 'nd2'
            self.file_reader = ND2Reader(file_name)
            self.is_closed = False 
        elif ('.tif' in file_name) or ('.tiff' in file_name):
            self.type = 'tif'
            self.file_reader = tifffile.TiffFile(file_name)
            self.is_closed = False 
        elif ('.czi' in file_name):
            self.type = 'czi'
            self.file_reader = CziFile(file_name)
            self.is_closed = False 
        else:
            print("Image format %s not recognized" % \
                os.path.splitext(file_name)[1])
            self.type = None
            self.is_closed = True 

    def get_shape(self):
        """
        returns
        -------
            (int, int, int), the y dimension, x dimension, and
            t dimension of the data
        
        """
        if self.is_closed:
            raise RuntimeError("Object is closed")
            
        if self.type == 'nd2':
            y_dim = self.file_reader.metadata['height']
            x_dim = self.file_reader.metadata['width']
            t_dim = self.file_reader.metadata['total_images_per_channel']
        elif self.type == 'tif' or self.type == 'czi':
            y_dim, x_dim = self.file_reader.pages[0].shape 
            t_dim = len(self.file_reader.pages)
        
        return (y_dim, x_dim, t_dim)

    def get_frame(self, frame_idx):
        """
        args
        ----
            frame_idx :  int

        returns
        -------
            2D ndarray, the corresponding frame

        """
        if self.is_closed:
            raise RuntimeError("Object is closed")
            
        if self.type == 'nd2':
            return self.file_reader.get_frame_2D(t = frame_idx)
        elif self.type == 'tif' or self.type == 'czi':
            return self.file_reader.pages[frame_idx].asarray()

    def min_max(self, frame_range=None):
        """
        args
        ----
            frame_range :  None or (int, int), the 
                first and last frames to check

        returns
        -------
            (int, int), the minimum and maximum pixel
                intensities in the stack

        """
        N, M, n_frames = self.get_shape()
        c_max, c_min = 0, 0

        if frame_range is None:
            frame_range = (0, n_frames)

        for frame_idx in range(frame_range[0], frame_range[1]):
            frame = self.get_frame(frame_idx)
            frame_min = frame.min()
            frame_max = frame.max()
            if frame_min < c_min:
                c_min = frame_min
            if frame_max > c_max:
                c_max = frame_max
        return c_min, c_max 

    def close(self):
        self.file_reader.close()
        self.is_closed = True 





