#!/usr/bin/env python
"""
mask.py -- apply binary masks to an SPT movie

"""
import os
import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.path import Path 
from scipy import interpolate 
from scipy.spatial import distance_matrix 
from tqdm import tqdm 

class MaskInterpolator(object):
    """
    Given a set of 2D masks, interpolate the masks between frames to 
    generate an approximation to the mask for intermediate frames.

    In more detail:

    We have a set of 2D masks defined in frames F0, F1, ..., Fn. Masks
    are defined as a ringlike, connected set of points. We wish to use
    the masks defined in frames F(i) and F(i+1) to estimate the shape
    of the mask in any intermediate frame, assuming the mask varies 
    smoothly/linearly between frame F(i) and F(i+1).

    Instantiation of the LinearMaskInterpolator() accomplishes this, 
    using either linear or spline interpolation. The resulting object
    can then be passed any frame index between F0 and Fn and will 
    generate the corresponding 2D mask.

    The resulting object can be passed a frame index, and will generate
    the corresponding 2D mask.

    init
    ----
        mask_edges      :   list of 2D ndarray of shape (n_points, 2),
                            the Y and X coordinates for each mask at 
                            each frame
        mask_frames     :   list of int, the frame indices corresponding
                            to each mask edge
        n_vertices      :   int, the number of vertices to use per 
                            interpolated mask
        interp          :   str, "linear" or "cubic", the type of 
                            interpolation to use. Note that at least 4 
                            masks are required for cubic spline interpolation.
        plot            :   bool, show a plot of the vertex matching between
                            interpolated masks during initialization, for QC

    methods
    -------
        __call__        :   determine whether each of a set of points lies
                            inside or outside the mask
        interpolate     :   generate an interpolated mask edge for an arbitrary
                            frame lying between the minimum and maximum frames
                            for this object

    """
    def __init__(self, mask_edges, mask_frames, n_vertices=101, 
        interp_kind="linear", plot=True):

        assert len(mask_edges) == len(mask_frames)
        assert interp_kind in ["linear", "cubic"]

        self.mask_edges = mask_edges 
        self.mask_frames = mask_frames 
        self.interp_kind = interp_kind
        self.n_frames = len(self.mask_frames)
        self.n_vertices = n_vertices 
        self.plot = plot 

        # If passed only a single frame, then the interpolator
        # always returns a simple static 2D shape
        self.static = (self.n_frames == 1)

        # Otherwise, generate interpolator objects to reconstruct
        # the mask for any other frame
        if not self.static:
            self._generate_mask_matches()
            self._generate_interpolators()

    def __call__(self, points, frame_indices, progress_bar=False):
        """
        Given a set of points, determine whether each point lies inside
        or outside the present mask.

        args
        ----
            points          :   2D ndarray of shape (n_points, 2), the 
                                YX coordinates for each point
            frame_indices   :   1D ndarray of shape (n_points,), the 
                                frame indices corresponding to each point
            progress_bar    :   bool, show a progress bar

        returns
        -------
            1D ndarray of shape (n_points,), dtype bool

        """
        assert points.shape[0] == frame_indices.shape[0]

        # Format as ndarray
        if isinstance(points, pd.DataFrame):
            points = np.asarray(points)
        if isinstance(frame_indices, pd.Series):
            frame_indices = np.asarray(frame_indices)

        unique_frames = np.unique(frame_indices)
        if progress_bar:
            unique_frames = tqdm(unique_frames)
        assignments = np.empty(points.shape[0])
        for frame_index in unique_frames:
            mask = Path(self.interpolate(frame_index), closed=True)
            in_frame = frame_indices == frame_index 
            assignments[in_frame] = mask.contains_points(points[in_frame,:])
        return assignments

    def interpolate(self, frame):
        """
        Interpolate the mask edges for a given frame index.

        args
        ----
            frame       :   int, the frame index

        returns
        -------
            2D ndarray of shape (self.n_vertices, 2), the YX
                coordinates for the points along the edge of 
                the mask

        """
        if self.static:
            return self.mask_edges[0]
        else:

            # Frame lies outside of interpolable range
            if not self._within_interpolation_range(frame):
                return self.mask_edges[:,-1,:]

            # Otherwise interpolate the mask
            result = np.empty((self.n_vertices, 2), dtype=np.float64)
            result[:,0] = [self.y_interpolators[i](frame) for i in range(self.n_vertices)]
            result[:,1] = [self.x_interpolators[i](frame) for i in range(self.n_vertices)]

            return result 

    def _within_interpolation_range(self, frame):
        """
        Return True if the frame lies inside the interpolation
        range for this MaskInterpolator instance.

        """
        return (frame >= self.mask_frames[0]) and (frame <= self.mask_frames[-1])

    def _generate_mask_matches(self):
        """
        Given the set of points that defines the edge of each mask
        used to instantiate this object, upsample the masks to the 
        same number of points and match each point with the points
        in other masks for subsequent interpolation.

        Resets the self.mask_edges attribute to the result.

        """
        # The total number of sets of mask edges to match 
        n_masks = len(self.mask_edges)

        # The final set of matched points, appropriate for 
        # interpolation
        result = np.zeros((self.n_vertices, n_masks, 2), dtype=np.float64)

        # Upsample each of the masks to the same number of points
        for j in range(n_masks):
            self.mask_edges[j] = upsample_2d_path(self.mask_edges[j],
                kind="cubic", n_vertices=self.n_vertices)

        result[:,0,:] = self.mask_edges[0]

        # For each sequential combination of masks, match the
        # interpolated mask points 
        for j in range(1, n_masks):
            result[:,j,:] = match_vertices(
                result[:,j-1,:],
                self.mask_edges[j],
                method="global",
                plot=self.plot
            )

        self.mask_edges = result 

    def _generate_interpolators(self):
        """
        For each point that defines the edge of this mask, generate Y
        and X interpolators that enable a mask to be reconstructed
        for a given image frame.

        """
        self.y_interpolators = []
        self.x_interpolators = []

        for vertex in range(self.n_vertices):

            # Generate the interpolator object for the y-index 
            I = interpolate.interp1d(self.mask_frames, self.mask_edges[vertex, :, 0],
                kind=self.interp_kind)
            self.y_interpolators.append(I)

            # Generate the interpolator object for the y-index 
            I = interpolate.interp1d(self.mask_frames, self.mask_edges[vertex, :, 1],
                kind=self.interp_kind)
            self.x_interpolators.append(I)

def upsample_2d_path(points, kind="cubic", n_vertices=101):
    """
    Upsample a 2D path by interpolation.

    args
    ----
        points      :   2D ndarray of shape (n_points, 2), the 
                        Y and X coordinates of each point in the 
                        path, organized sequentially
        kind        :   str, the kind of spline interpolation
        n_vertices  :   int, the number of points to use in the 
                        upsampled path

    returns
    -------
        2D ndarray of shape (n_vertices, 2), the upsampled 
            path

    """
    P = np.concatenate((points, np.array([points[0,:]])), axis=0)
    t = np.arange(P.shape[0])
    fy = interpolate.interp1d(t, P[:,0], kind=kind)
    fx = interpolate.interp1d(t, P[:,1], kind=kind)
    result = np.empty((n_vertices, 2), dtype=np.float64)
    new_t = np.linspace(0, t.max(), n_vertices)
    result[:,0] = fy(new_t)
    result[:,1] = fx(new_t)
    return result 

def shoelace(points):
    """
    Shoelace algorithm for computing the oriented area of a 2D
    polygon. This area is positive when the points that define
    the polygon are arranged counterclockwise, and negative
    otherwise.

    args
    ----
        points      :   2D ndarray, shape (n_points, 2), the 
                        vertices of the polygon

    returns
    -------
        float, the oriented volume of the polygon defined by 
            *points*

    """
    return ((points[1:,0] - points[:-1,0]) * \
        (points[1:,1] + points[:-1,1])).sum()

def circshift(points, shift):
    """
    Circularly shift a set of points.

    args
    ----
        points      :   ndarray of shape (n_points, D), the 
                        D-dimensional coordinates of each point 
        shift       :   int, the index of the new starting point

    returns
    -------
        2D ndarray of shape (n_points, D), the same points
            but circularly shifted

    example
    -------
        points_before = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ])

        points_after = circshift(points_before, 1)

        points_after -> np.array([
            [3, 4],
            [5, 6],
            [1, 2]
        ])

    """
    out = np.empty(points.shape, dtype=points.dtype)
    n = out.shape[0]
    shift = shift % n 
    if len(points.shape) == 1:
        out[:n-shift] = points[shift:]
        out[n-shift:] = points[:shift]
    elif len(points.shape) == 2:
        out[:n-shift,:] = points[shift:,:]
        out[n-shift:,:] = points[:shift,:]
    return out 

def match_vertices(vertices_0, vertices_1, method="closest", plot=False):
    """
    Given two polygons with the same number of vertices, match
    each vertex in the first polygon with the "closest" vertex
    in the second polygon.

    "closest" is in quotation marks here because, before matching,
    we align the two polygons by their mean position, so that the
    same match is returned regardless of whole-polygon shifts.

    args
    ----
        vertices_0      :   2D ndarray, shape (n_points, 2), the 
                            YX coordinates for the vertices of the 
                            first polygon
        vertices_1      :   2D ndarray, shape (n_points, 2), the 
                            YX coordinates for the vertices of the 
                            second polygon
        method          :   str, the method to use to match vertices.
                            "closest": use the closest point between
                            the two masks as the anchor point. 
                            "global": use the permutation that minimizes
                            the total distance between the two sets
                            of vertices.
        plot            :   bool, show the result

    returns
    -------
        2D ndarray, shape (n_points, 2), the vertices of the 
            second polygon circularly permuted to line them up
            with the matching vertex in the first polygon

    """
    assert vertices_0.shape == vertices_1.shape
    assert method in ["closest", "global"], "method must be either 'closest' or 'global'"

    # The final assignments
    indices_1 = np.arange(vertices_1.shape[0])

    # Deal only with the positions of each vertex relative to the 
    # respective mean
    P0 = vertices_0 - vertices_0.mean(axis=0)
    P1 = vertices_1 - vertices_1.mean(axis=0)

    # Make sure the points both proceed in the same direction (CW or CCW)
    if shoelace(P0) * shoelace(P1) < 0:
        P1 = P1[::-1,:]
        indices_1 = indices_1[::-1]

    # Align masks by simply looking for two vertices that are closest
    if method == "closest":
        distances = distance_matrix(P0, P1)
        m = np.argmin(distances.ravel())
        y, x = m // P0.shape[0], m % P0.shape[0]
        shift = (x - y) % P0.shape[0]

    # Align masks by minimizing the sum of the distances between all
    # vertices for all possible matches
    elif method == "global":
        curr = 0
        ss = np.inf 
        for x in range(P1.shape[0]):
            shift_P1 = circshift(P1, x)
            tot_dist = ((P0 - shift_P1)**2).sum()
            if tot_dist < ss:
                curr = x 
                ss = tot_dist 
        shift = curr 

    # Align the masks
    indices_1 = circshift(indices_1, shift)
    vertices_1 = vertices_1[indices_1, :]

    # Show the resulting set of vertex matches, if desired
    if plot:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.scatter(vertices_0[:,0], vertices_0[:,1], cmap="viridis",
            c=np.arange(vertices_0.shape[0]))
        ax.scatter(vertices_1[:,0], vertices_1[:,1], cmap="viridis",
            c=np.arange(vertices_1.shape[0]))
        for j in range(vertices_0.shape[0]):
            ax.plot([vertices_0[j,0], vertices_1[j,0]], [vertices_0[j,1], vertices_1[j,1]], 
                color="k", linestyle='-')
        ax.set_aspect('equal')
        ax.set_title("Mask alignment")
        plt.show()

    return vertices_1 

