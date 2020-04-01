"""
utils.py

"""
# Numerics
import numpy as np 

# Special functions for PSF definitions
from scipy.special import erf 

# Get the args to a function
import inspect 

# Nd image analysis
from scipy import ndimage as ndi 

# Warning control
import warnings 

# Random 8-bit color indices 
n_colors = 1028
colors_8bit = np.random.randint(256, size=(1028, 4),
    dtype='uint8')

# 
# BASIC UTILITIES ON NDARRAYS AND FLOATS
#
def str_to_bool(S):
    """
    Try to read a boolean value from a string.

    """
    if isinstance(S, bool):
        return S
    elif isinstance(S, str):
        if (S=='True') or (S=='true') or (S=='T') or (S=='t'):
            return True 
        elif (S=='False') or (S=='false') or (S=='F') or (S=='f'):
            return False 
        else:
            return S 

def set_neg_to_zero(ndarray):
    """
    Set all negative values in a 
    numpy.ndarray to zero.

    """
    ndarray[ndarray<0.0] = 0.0
    return ndarray 

def threshold_image(image, t):
    """
    Return a binary image based on an 
    intensity threshold.

    """
    return (image>=t).astype('uint8')

def stable_divide_float(n, d, inf=0.0):
    """
    Divide n by d if d is nonzero,
    else return *inf*.

    """
    if d == 0.0:
        return inf 
    else:
        return n/d

def stable_divide(N, D, inf=1.0):
    """
    Divide two arrays, replacing any divide-by-zero
    errors with the value *inf*. 

    args
    ----
        N : numerator matrix
        D : denominator matrix, assumed positive
        inf : the value to replace any divide-by-
            zero errors with

    returns
    -------
        2D ndarray

    """
    # Make sure the shapes coincide
    assert N.shape == D.shape 

    # Format output
    result = np.full(N.shape, inf, dtype='float64')

    # Determine where the denominator is zero
    nonzero = D > 0.0

    # Perform division
    result[nonzero] = N[nonzero] / D[nonzero]
    return result

def ring_mean(img):
    """
    Take the mean of the outer ring of pixels
    in a 2D ndarray.

    """
    return np.concatenate([
        img[0,1:],
        img[1:,-1],
        img[-1,:-1],
        img[:-1,0]
    ]).mean()

def ring_var(img, ddof=0):
    """
    Take the variance of the outer ring of pixels
    in a 2D ndarray.

    """
    return np.concatenate([
        img[0,1:],
        img[1:,-1],
        img[-1,:-1],
        img[:-1,0]
    ]).var(ddof=ddof)

def invert_hessian(H, ridge=-0.0001):
    """
    Try to invert a Hessian using regularization
    with a ridge term. The ridge term is increased
    in magnitude until inversion can be performed.

    The ridge term is by default negative, because
    this is commonly used to stabilize the Hessian
    which converges to an ML estimate when negative
    definite.

    args
    ----
        H : 2D ndarray of shape (n, n)
        ridge : float, initial ridge term  

    returns
    -------
        2D ndarray of shape (n, n)

    """
    D = np.diag(np.ones(H.shape[0]) * ridge)
    while 1:
        try:
            H_inv = np.linalg.inv(H-D)
            return H_inv 
        except (ZeroDivisionError, np.linalg.linalg.LinAlgError):
            D *= 10 
            continue 

def pinv(M):
    """
    Compute the Moore-Penrose pseudoinverse of a 
    square matrix.

    args
    ----
        M : 2D ndarray, square

    returns
    -------
        2D ndarray of shape M.shape 

    """
    return np.linalg.inv(M.T @ M) @ M.T 

def get_pivots(M):
    """
    Find the pivots (1D ndarray) of a square
    matrix using Gaussian elimination.

    """
    A = M.copy()
    n = A.shape[0]

    # For each column
    for c in range(n-1):

        # If we encounter a zero pivot, let the program know
        if M[c,c] == 0.0:
            raise ZeroDivisionError

        # For each row 
        for r in range(c+1, n):
            A[r,:] = A[r,:] - A[c,:]*A[r,c]/A[c,c]

    # Return the diagonal, giving the pivots
    return np.diagonal(A)

def get_slice(img, hw, pos):
    """
    Get a small square subregion centered at *pos*.
    Necessarily, will always return windows of 
    odd size.

    args
    ----
        img : 2D ndarray
        hw : int, half-window size 
        pos : (int, int), center

    returns
    -------
        2D ndarray of shape (2*hw+1, 2*hw+1)

    """
    return img[pos[0]-hw:pos[0]+hw+1, 
        pos[1]-hw:pos[1]+hw+1]

def get_gaussian_kernel(w, k):
    """
    Generate a Gaussian detection kernel of window size
    *w* and sigma *k*.

    args
    ----
        w : int
        k : float

    returns
    -------
        2D ndarray, shape (w, w)

    """
    var2 = 2 * (k**2)
    g = np.exp(-((np.indices((w, w))-(w-1)/2)**2\
        ).sum(axis=0) / var2)
    g = g / g.sum()
    return g 

def detections_inside_edge(yx_df, frame_size,
    edge_size):
    """
    Given a dataframe with detections, return
    an array that is False when the corresponding
    detection is too close to the edge of a 
    frame.

    args
    ----
        yx_df : pandas.DataFrame with columns
            `yd` and `xd` indicating detection
            coords
        frame_size : (int, int)
        edge_size : int

    returns
    -------
        1D ndarray, dtype bool

    """
    return (yx_df['yd']>=edge_size) & \
        (yx_df['yd']<frame_size[0]-edge_size) & \
        (yx_df['xd']>=edge_size) & \
        (yx_df['xd']<frame_size[1]-edge_size)

def remove_edge_detections(yx_df, frame_size,
    edge_size):
    """
    Remove all rows with detections (`yd`, `xd`) too
    close to the edge.

    """
    return yx_df[detections_inside_edge(yx_df,
        frame_size, edge_size)]

def detection_inside_edge(yx_pos, frame_size,
    edge_size):
    """
    Return False if the point is too close
    to the edge of a frame.

    args
    ----
        yx_pos : (int, int), point position
        frame_size : (int, int)
        edge_size : int

    returns
    -------
        bool

    """
    return (yx_pos[0]>=edge_size) and \
        (yx_pos[0]<frame_size[0]-edge_size) and \
        (yx_pos[1]>=edge_size) and \
        (yx_pos[1]<frame_size[1]-edge_size)

# 
# LOW-LEVEL DETECTION AND CONVOLUTION UTILITIES
#
def remove_edge_cases(detections, w, imshape):
    """
    Remove all detections that are within *w*
    pixels of the edge of a frame.

    args
    ----
        detections : pandas.DataFrame
        w : int
        imshape : (int, int)

    returns
    -------
        pandas.DataFrame

    """
    pass

def get_slice(img, hw, pos):
    """
    Get a small square subregion centered at *pos*.
    Necessarily, will always return windows of 
    odd size.

    args
    ----
        img : 2D ndarray
        hw : int, half-window size 
        pos : (int, int), center

    returns
    -------
        2D ndarray of shape (2*hw+1, 2*hw+1)

    """
    return img[pos[0]-hw:pos[0]+hw+1, 
        pos[1]-hw:pos[1]+hw+1]

def upsample(img, upsampling=2):
    """
    Return an upsampled version of an image.

    args
    ----
        img : 2D ndarray
        upsampling : int

    returns
    -------
        2D ndarray

    """
    if len(img.shape) == 2:
        N, M = img.shape 
        N_up = N * upsampling 
        M_up = M * upsampling
        result = np.zeros((N_up, M_up), dtype=img.dtype)
        for i in range(upsampling):
            for j in range(upsampling):
                result[i::upsampling, j::upsampling] = img 
    elif len(img.shape) == 3:
        N, M, n_channels = img.shape 
        N_up = N * upsampling 
        M_up = M * upsampling
        result = np.zeros((N_up, M_up, n_channels), dtype=img.dtype)
        for i in range(upsampling):
            for j in range(upsampling):
                result[i::upsampling, j::upsampling, :] = img 
    return result 

def upsample_overlay_two_color(
    img, 
    white_positions,
    red_positions, 
    upsampling=2,
    crosshair_len=8,
):
    """
    Overlay some positions in white and some in red.

    args
    ----
        img : 2D ndarray of dtype uint8, or 3D ndarray in
            RGBA format (YXC)
        white_positions : 2D ndarray of shape (n_points, 2)
        red_positions : 2D ndarray of shape (n_points, 2)
        upsampling : int
        crosshair_len : int

    returns
    -------
        3D ndarray (YXC), RGBA result

    """
    # Expand the image 
    if len(img.shape) == 2:
        N, M = img.shape 
        N_up = N * upsampling 
        M_up = M * upsampling
        I = np.zeros((N_up, M_up, 4), dtype='uint8')
        for i in range(upsampling):
            for j in range(upsampling):
                I[i::upsampling, j::upsampling, 3] = img 

    elif len(img.shape) == 3:
        N, M, n_channels = img.shape 
        N_up = N * upsampling 
        M_up = M * upsampling
        I = np.zeros((N_up, M_up, 4), dtype='uint8')
        for i in range(upsampling):
            for j in range(upsampling):
                I[i::upsampling, j::upsampling, :] = img 

    # Round positions to the nearest integer
    P_whi = (white_positions*upsampling).astype('int64')
    P_red = (red_positions*upsampling).astype('int64')

    # Do the overlay 
    for j in range(-crosshair_len, crosshair_len+1):

        # White positions 

        # Extend crosshair in y direction
        PY = P_whi[:,0] + j 
        PX = P_whi[:,1]
        inside = (PY>=0) & (PY<N_up) & (PX>=0) & (PX<M_up)
        I[PY[inside], PX[inside], 3] = 0

        # Extend crosshair in x direction
        PY = P_whi[:,0]
        PX = P_whi[:,1] + j 
        inside = (PY>=0) & (PY<N_up) & (PX>=0) & (PX<M_up)
        I[PY[inside], PX[inside], 3] = 0

        # Red positions

        # Extend crosshair in y direction
        PY = P_red[:,0] + j 
        PX = P_red[:,1] 
        inside = (PY>=0) & (PY<N_up) & (PX>=0) & (PX<M_up)
        I[PY[inside], PX[inside], 3] = 255
        I[PY[inside], PX[inside], 0] = 255 

        # Extend crosshair in x direction
        PY = P_red[:,0]
        PX = P_red[:,1] + j 
        inside = (PY>=0) & (PY<N_up) & (PX>=0) & (PX<M_up)
        I[PY[inside], PX[inside], 3] = 255
        I[PY[inside], PX[inside], 0] = 255

    return I 

def upsample_overlay_trajs(img, vmax=None, vmin=None,
    pos=None, traj_indices=None, u=3, crosshair_len=9,
    crosshair_type='+'):
    """
    Upsample an image and overlay a set of 
    localizations as crosshairs (+), coloring
    them by their trajectory index.

    Return the upsampled image as 8-bit RGBA.

    args
    ----
        img : 2D ndarray
        pos : 2D ndarray of shape (n_locs, 2),
            the YX positions in this frame (in
            pixels)
        traj_indices : 1D ndarray of shape 
            (n_locs), the trajectory indices
        u : int, upsampling factor 

    returns
    -------
        3D ndarray, 8-bit RGBA overlay

    """
    # Rescale and convert to 8-bit
    if vmax is None:
        vmax = img.max()
    if vmin is None:
        vmin = img.min()
    if vmin == vmax:
        img = np.zeros(img.shape, dtype='uint8')
    else:
        img = 255.0*(img.astype('float64')-vmin)/(vmax-vmin)
        img[img>255.0] = 255.0
        img[img<0.0] = 0.0
        img = img.astype('uint8')

    # Invert the color scheme
    img = 255 - img 

    # Upsample
    N, M = img.shape 
    N_up = N * u 
    M_up = M * u
    I = np.zeros((N_up, M_up, 4), dtype='uint8')
    for i in range(u):
        for j in range(u):
            I[i::u,j::u,3] = img

    if not pos is None:

        # Convert positions to upsampled pixels, and 
        # round to nearest integer
        P = (pos * u).astype('int64')

        # Get the colors of each spot 
        colors = colors_8bit[traj_indices%n_colors,:3]

        # Do the overlay 
        if crosshair_type == '+':
            for j in range(-crosshair_len, crosshair_len+1):

                # Extend crosshair in y direction
                PY = P[:,0] + j 
                PX = P[:,1]
                inside = (PY>=0) & (PY<N_up) & (PX>=0) & (PX<M_up)
                I[PY[inside], PX[inside], :3] = colors[inside]
                I[PY[inside], PX[inside], 3] = 255

                # Extend crosshair in x direction
                PY = P[:,0]
                PX = P[:,1] + j 
                inside = (PY>=0) & (PY<N_up) & (PX>=0) & (PX<M_up)
                I[PY[inside], PX[inside], :3] = colors[inside]
                I[PY[inside], PX[inside], 3] = 255
        elif crosshair_type == 'o':
            coord_shifts = [(5,0), (4,0), (4,1), (4,2), (4,3), 
                (3,3), (3,4), (2,4), (1,4), (0,4), (0,5),
                (-1,4), (-2,4), (-3,4), (-3,3), (-4,3), (-4,2),
                (-4,1), (-4,0), (-5,0), (-4,-1), (-4,-2),
                (-4,-3), (-3,-3), (-3,-4), (-2,-4), (-1,-4),
                (0,-4), (0,-5), (1,-4), (2,-4), (3,-4), 
                (3,-3), (4,-3), (4,-2), (4,-1)]

            for i, j in coord_shifts:
                PY = P[:,0] + i 
                PX = P[:,1] + j 
                inside = (PY>=0) & (PY<N_up) & (PX>=0) & (PX<M_up)
                I[PY[inside], PX[inside], :3] = colors[inside]
                I[PY[inside], PX[inside], 3] = 255

    return I 

def upsample_overlay_trajs_history(img, vmax=None, vmin=None,
    pos_history=None, traj_indices=None, u=3, crosshair_len=9,
    crosshair_type='+'):
    """
    Upsample an image and overlay a set of 
    localizations as crosshairs (+), coloring
    them by their trajectory index.

    This version will plot a full trajectory,
    not a single localization.

    Return the upsampled image as 8-bit RGBA.

    args
    ----
        img : 2D ndarray
        pos_history : list of 2D ndarray, the 
            YX positions of each trajectory over
            some number of frames
        pos : 2D ndarray of shape (n_locs, 2),
            the YX positions in this frame (in
            pixels)
        traj_indices : 1D ndarray of shape 
            (n_locs), the trajectory indices
        u : int, upsampling factor 

    returns
    -------
        3D ndarray, 8-bit RGBA overlay

    """
    # Rescale and convert to 8-bit
    if vmax is None:
        vmax = img.max()
    if vmin is None:
        vmin = img.min()
    if vmin == vmax:
        img = np.zeros(img.shape, dtype='uint8')
    else:
        img = 255.0*(img.astype('float64')-vmin)/(vmax-vmin)
        img[img>255.0] = 255.0
        img[img<0.0] = 0.0
        img = img.astype('uint8')

    # Invert the color scheme
    img = 255 - img 

    # Upsample
    N, M = img.shape 
    N_up = N * u 
    M_up = M * u
    I = np.zeros((N_up, M_up, 4), dtype='uint8')
    for i in range(u):
        for j in range(u):
            I[i::u,j::u,3] = img

    # Do an overlay
    if not pos_history is None and len(pos_history) > 0:

        # Get the colors corresponding to each trajectory
        colors = colors_8bit[traj_indices%n_colors,:3]

        # Draw a crosshairs at the last positions of each 
        # trajectory
        P = np.asarray([pos[-1,:] for pos in pos_history])

        # Convert positions to upsampled pixels, and 
        # round to nearest integer
        P = (P * u).astype('int64')

        if crosshair_type == '+':
            for j in range(-crosshair_len, crosshair_len+1):

                # Extend crosshair in y direction
                PY = P[:,0] + j 
                PX = P[:,1]
                inside = (PY>=0) & (PY<N_up) & (PX>=0) & (PX<M_up)
                I[PY[inside], PX[inside], :3] = colors[inside]
                I[PY[inside], PX[inside], 3] = 255

                # Extend crosshair in x direction
                PY = P[:,0]
                PX = P[:,1] + j 
                inside = (PY>=0) & (PY<N_up) & (PX>=0) & (PX<M_up)
                I[PY[inside], PX[inside], :3] = colors[inside]
                I[PY[inside], PX[inside], 3] = 255
        elif crosshair_type == 'o':
            coord_shifts = [(5,0), (4,0), (4,1), (4,2), (4,3), 
                (3,3), (3,4), (2,4), (1,4), (0,4), (0,5),
                (-1,4), (-2,4), (-3,4), (-3,3), (-4,3), (-4,2),
                (-4,1), (-4,0), (-5,0), (-4,-1), (-4,-2),
                (-4,-3), (-3,-3), (-3,-4), (-2,-4), (-1,-4),
                (0,-4), (0,-5), (1,-4), (2,-4), (3,-4), 
                (3,-3), (4,-3), (4,-2), (4,-1)]

            for i, j in coord_shifts:
                PY = P[:,0] + i 
                PX = P[:,1] + j 
                inside = (PY>=0) & (PY<N_up) & (PX>=0) & (PX<M_up)
                I[PY[inside], PX[inside], :3] = colors[inside]
                I[PY[inside], PX[inside], 3] = 255

        # For each trajectory, draw lines indicating
        # its past movement
        for traj_idx, pos in enumerate(pos_history):

            if pos.shape[0] > 1:

                # Convert positions to upsampled pixels, and 
                # round to nearest integer
                P = (pos * u).astype('int64')

                # Get this spot's color
                color = colors[traj_idx]

                # Get a line with Bresenham's algorithm
                for j in range(1, pos.shape[0]):
                    line = np.asarray(bresenham(P[j,:], P[j-1,:]))
                    I[line[:,0], line[:,1], :3] = color 
                    I[line[:,0], line[:,1], 3] = 255 

    return I 

def bresenham(p0, p1):
    """
    Bresenham's algorithm for drawing a line in 
    terms of discrete pixels.

    args
    ----
        p0 : (int, int), (x, y) for first point
        p1 : (int, int), (x, y) for second point

    returns
    -------
        2D ndarray of shape (n_points, 2), the 
            YX coords of each point

    """
    x0, y0 = p0
    x1, y1 = p1 
    dx = abs(x1-x0)
    dy = abs(y1-y0)
    if x0 < x1:
        sx = 1
    else:
        sx = -1 
    if y0 < y1:
        sy = 1 
    else:
        sy = -1 
    err = dx - dy 

    cx = x0
    cy = y0

    result = [(cx, cy)]

    while (x0 != x1) and (y0 != y1):
        e2 = 2 * err 
        if e2 > -dy:
            err = err - dy 
            x0 = x0 + sx 
        if e2 < dx:
            err = err + dx 
            y0 = y0 + sy 

        result.append((x0, y0))

    return result 

def overlay_spots_rgba(img, positions, 
    channel=0, crosshair_len=4):
    """
    Make a copy of an RGBA image and write crosshairs
    over it at a set of defined positions in a 
    particular color.

    args
    ----
        image : 3D ndarray, RGBA image
        positions : 2D ndarray of shape
            n_points, 2), the positions
            of the points
        channel : int, RGB color index 
        crosshair_len : int, size of the 
            crosshairs

    returns
    -------
        3D ndarray, copy of the image with
            the overlayed crosshairs

    """
    I = img.copy()

    # If no positions are passed, return the same image 
    if (positions.shape[0] == 0) or \
        (len(positions.shape) < 2):
        return I 

    # Round positions to the nearest integer
    P = positions.astype('int64')

    I_max = I.max()
    N, M, n_color = I.shape

    for j in range(-crosshair_len, crosshair_len+1):

        # Extend crosshair in y direction
        PY = P[:,0] + j 
        PX = P[:,1]
        inside = (PY>=0) & (PY<N) & (PX>=0) & (PX<M)
        I[PY[inside], PX[inside], 3] = 255 
        if channel == 3:
            pass
        else:
            I[PY[inside], PX[inside], channel] = 255 

        # Extend crosshair in x direction
        PY = P[:,0]
        PX = P[:,1] + j 
        inside = (PY>=0) & (PY<N) & (PX>=0) & (PX<M)
        I[PY[inside], PX[inside], 3] = 255
        if channel == 3:
            pass 
        else:
            I[PY[inside], PX[inside], channel] = 255

    return I 

def overlay_spots(image, positions,
    crosshair_len=4):
    """
    Make a copy of an image and write crosshairs
    over it at a set of defined positions.

    args
    ----
        image : 2D ndarray, the image 
        positions : 2D ndarray of shape
            n_points, 2), the positions
            of the points
        crosshair_len : int, size of the 
            crosshairs

    returns
    -------
        2D ndarray, copy of the image with
            the overlayed crosshairs

    """
    I = image.copy()

    # If no positions are passed, return the same image
    if (positions.shape[0] == 0) or \
        (len(positions.shape) < 2):
        return I 

    I_max = I.max()
    N, M = I.shape 

    for j in range(-crosshair_len, crosshair_len+1):

        # Extend crosshair in y direction
        PY = positions[:,0] + j 
        PX = positions[:,1]
        inside = (PY>=0) & (PY<N) & (PX>=0) & (PX<M)
        I[PY[inside], PX[inside]] = I_max 

        # Extend crosshair in x direction
        PY = positions[:,0]
        PX = positions[:,1] + j 
        inside = (PY>=0) & (PY<N) & (PX>=0) & (PX<M)
        I[PY[inside], PX[inside]] = I_max 

    return I 

def label_binary_spots(img_bin, img_int=None):
    """
    Find the centers of contiguous nonzero objects
    in a binary image, returning the coordinates
    of the spots as a 2D ndarray.

    If *img_int* is passed, then the coordinates
    are the nearest pixels to the centroid of 
    *img_int*. Otherwise the coordinates are the
    nearest pixels to the mean position of the 
    binary spot.

    args
    ----
        img_bin : 2D ndarray, binary spot image
        img_int : 2D ndarray, the intensities
            for (optional) centroid calculations

    returns
    -------
        2D ndarray of shape (n_spots, 2) and
            dtype int64, the YX coordinates of 
            each spot

    """
    img_lab, N = ndi.label(img_bin)
    index = np.arange(1,N+1)
    if img_int is None:
        positions = np.asarray([ndi.center_of_mass(
            img_bin, labels=img_lab, index=index)])
    else:
        positions = np.asarray(ndi.center_of_mass(
            img_int, labels=img_lab, index=index))
    return positions.astype('int64')

def zero_pad(img, H, W, mode='ceil'):
    """
    Pad a kernel with zeroes for subsequent
    convolution.

    args
    ----
        img : 2D ndarray, kernel
        H : int, desired height
        W : int, desired width
        mode : adjustment for nearest-pixel
            effects. 'ceil' is consisent
            with scipy.ndimage.uniform_filter

    returns
    -------
        2D ndarray

    """
    H_in, W_in = img.shape
    out = np.zeros((H, W))
    if mode == 'ceil':
        hc = np.ceil(H / 2 - H_in / 2).astype(int)
        wc = np.ceil(W / 2 - W_in / 2).astype(int)
    elif mode == 'floor':
        hc = np.floor(H / 2 - H_in / 2).astype(int)
        wc = np.floor(W / 2 - W_in / 2).astype(int)
    out[hc : hc + H_in, wc : wc + W_in] = img
    return out

def rfftconvolve(image, kernel):
    """
    Convolve an image with a kernel.

    args
    ----
        image : 2D ndarray
        kernel : 2D ndarray, equal or smaller
                in size to *image*

    returns
    -------
        2D ndarray

    """
    image_rft = np.fft.rfft2(image)
    kernel_rft = np.fft.rfft2(zero_pad(kernel, 
        *image.shape))
    return np.fft.fftshift(np.fft.irfft2(image_rft* \
        kernel_rft))

#
# LOW-LEVEL LOCALIZATION UTILITIES
#
def _loc_kwargs_valid(**kwargs):
    """
    Return True if the keyword argument list
    contains the essential localization 
    keywords `y0`, `x0`, `I`, and `bg, which
    are the components of the 2D Gaussian PSF
    model.

    """
    return all([i in kwargs.keys() for i in \
        ['y0', 'x0', 'I', 'bg']])

def rescale_img(img, camera_offset=0.0, camera_gain=1.0):
    """
    Subtract offset and divide out gain of 
    a camera.

    """
    if (camera_offset!=0.0) and (camera_gain!=1.0):
        return (img-camera_offset)/camera_gain
    else:
        return img 

def fit_is_sane(fit_pars, w, max_I=10000, border=2):
    """
    Check whether a set of fit parameters make physical
    sense - i.e. they're inside the fitting window.

    args
    ----
        fit_pars : dict with 'y0', 'x0', 'I', and
            'bg', or 1D ndarray of size 4 with the
            same parameters
        max_I : float, max permissible intensity

    returns
    -------
        bool

    """
    b0 = border 
    b1 = w - border 
    if isinstance(fit_pars, dict):
        return (fit_pars['y0']>=b0) and (fit_pars['y0']<=b1) and \
            (fit_pars['x0']>=b0) and (fit_pars['x0']<=b1) and \
            (fit_pars['I']>=0) and (fit_pars['I']<=max_I) and \
            (fit_pars['bg']>=0)
    else:
        return (fit_pars[0]>=b0) and (fit_pars[0]<=b1) and \
            (fit_pars[1]>=b0) and (fit_pars[1]<=b1) and \
            (fit_pars[2]>=0) and (fit_pars[2]<=max_I) and \
            (fit_pars[3]>=0)

def psf_int_1d(X, x0, sigma=1.0):
    """
    Return a 1D component of a Gaussian integrated
    across discrete pixels, used in the definition
    of the 2D Gaussian PSF.

    args
    ----
        X : ndarray, positions in this axis at which
            to evaluate the PSF
        x0 : float, the (proposed) center of the 
            PSF in this dimension
        sigma : float, width of Gaussian

    returns
    -------
        ndarray of shape X.shape, dtype float64

    """
    sqrt_var = np.sqrt(2*(sigma**2))
    return 0.5*(erf((X-x0+0.5)/sqrt_var) - \
        erf((X-x0-0.5)/sqrt_var))

def psf_int(Y, X, y0, x0, I, sigma=1.0):
    """
    Return a 2D Gaussian PSF integrated on
    discrete pixels of unit size.

    The inspiration is Smith C et al. Nat Methods 2010.

    args
    ----
        Y, X : 2D ndarrays, y and x indices of
            each pixel
        y0, x0 : floats, PSF centers in y and x
        I  : float, PSF integrated intensity
        sigma : float, width of Gaussian PSF

    returns
    -------
        2D ndarray of shape Y.shape, dtype float64

    """
    return I*psf_int_1d(Y, y0, sigma=sigma)* \
        psf_int_1d(X, x0, sigma=sigma)

def eval_psf_int(Y, X, sigma=1.0, **kwargs):
    """
    Evaluate a full PSF model, including BG.
    This function is designed to be passed
    kwargs from a variety of upstream functions,
    and finds the keywords necessary to construct
    the Gaussian. Other keywords present in 
    kwargs are ignored.

    quot.localize is designed such that any
    localization method must return at minimum the
    parameters necessary to run this function
    (`y0`, `x0`, `I`, and `bg`).

    args
    ----
        Y, X : 2D ndarrays, y and x indices of 
            each pixel
        sigma : float, Gaussian width 
        kwargs : must contain 'y0', 'x0', 'I',
            and 'bg'

    returns
    -------
        2D ndarray, the Gaussian PSF model 

    """
    assert _loc_kwargs_valid(**kwargs)
    return psf_int(Y, X, kwargs['y0'], kwargs['x0'],
        kwargs['I'], sigma=sigma) + kwargs['bg']

def psf_point(w, k=1.0):
    """
    Return the values of a 2D Gaussian PSF with
    sigma *k* evaluated at the center of each pixel.
    The Gaussian is centered in a window of
    shape (w, w) and ea

    Return a 2D Gaussian PSF of sigma *k* centered
    in a shape (w, w) window. The intensity at 
    each pixel is equal to the value of the Gaussian
    evaluated at the center of that pixel.

    This PSF model differs from psf_int() in
    that the Gaussian evaluated at the centers
    of each pixel is not equivalent to the Gaussian
    integrated across each pixel.

    Normalized to sum to 1. 

    args
    ----
        w : int, window size
        k : float, kernel width

    returns
    -------
        2D ndarray, dtype float64, shape (w, w)

    """
    result = np.exp(-((np.indices((w, w)) - \
        (w-1)/2)**2).sum(axis=0) / (2*(k**2)))
    return result / result.sum()

def eval_psf_point(Y, X, sigma=1.0, **kwargs):
    """
    Evaluate a full pointwise Gaussian PSF model,
    including BG. This function is designed to be 
    passed kwargs from a variety of upstream functions,
    and finds the keywords necessary to construct
    the Gaussian. Other keywords present in the 
    kwargs are ignored.

    quot.localize is designed such that any
    localization method must return at minimum the
    parameters necessary to run this function
    (`y0`, `x0`, `I`, and `bg`).

    args
    ----
        Y, X : 2D ndarrays, y and x indices of 
            each pixel
        sigma : float, Gaussian width 
        kwargs : must contain 'y0', 'x0', 'I',
            and 'bg'

    returns
    -------
        2D ndarray, the Gaussian PSF model 

    """
    assert _loc_kwargs_valid(**kwargs)
    return kwargs['I']*np.exp(-((Y-kwargs['y0'])**2 + \
        (X-kwargs['x0'])**2) / (2*(sigma**2))) / \
        (2*np.pi*(sigma**2)) + kwargs['bg']

def estimate_intensity(psf_img, y0, x0, bg,
    sigma=1.0, max_I=10000):
    """
    Given an image and a proposed center for
    a PSF, estimate the intensity of the pixel
    by evaluated a PSF model at the brightest
    pixel and solving for the intensity.

    This is a very common way to get the intensity
    for the noniterative localization methods,
    and a common first guess for the iterative
    methods.

    args
    ----
        psf_img : 2D ndarray
        y0, x0 : floats, proposed center of PSF
        bg : float, proposed BG of PSF 
        sigma : float, Gaussian PSF width 
        max_I : float, the maximum permissible
            value for the intensity. The intensity
            tends to diverge sometimes. If the
            estimated value for I exceeds this 
            value, fall back to a different method.

    returns
    -------
        float, the estimated intensity `I` for
            the PSF model, equal to the PSF
            integrated above background

    """
    # Get the maximum intensity pixel
    ym, xm = np.unravel_index(np.argmax(psf_img), 
        psf_img.shape)

    # Evaluate intensity estimate 
    estimate = stable_divide_float(
        psf_img[ym,xm] - bg,
        psf_int_1d(ym, y0, sigma=sigma) * \
            psf_int_1d(xm, x0, sigma=sigma),
        inf=np.nan, 
    )

    # Check whether it's crazy, and if so,
    # guess by a different method
    if (estimate < 0) or (estimate > max_I):
        estimate = (psf_img-bg).sum()

    return estimate 


def estimate_snr(psf_img, amp):
    """
    Estimate the signal-to-noise ratio given a 
    PSF image and a signal amplitude.

    Here, 

        SNR := amplitude^2 / noise variance

    args
    ----
        psf_img : 2D ndarray
        amp : float

    returns
    -------
        float, SNR estimate

    """
    return stable_divide_float(
        amp**2, ring_var(psf_img, ddof=1),
        inf=np.inf)

def amp_from_I(I, sigma=1.0):
    """
    Given a 2D Gaussian PSF, return the PSF
    peak amplitude given the intensity `I`.

    `I` is equal to the PSF integrated above
    background, while `amp` is equal to the PSF
    evaluated at its maximum.

    args
    ----
         I : float, intensity estimate
         sigma : float, width of Gaussian

    returns
    -------
        float, amplitude estimate

    """
    return I / (2*np.pi*(sigma**2))

def rs(psf_image):
    """
    Localize the center of a PSF using the radial 
    symmetry method.

    Originally conceived by the criminally underrated
    Parasarathy R Nature Methods 9, pgs 724–726 (2012).

    args
    ----
        psf_image : 2D ndarray, PSF subwindow

    returns
    -------
        float y estimate, float x estimate

    """
    # Get the size of the image frame and build
    # a set of pixel indices to match
    N, M = psf_image.shape
    N_half = N // 2
    M_half = M // 2
    ym, xm = np.mgrid[:N-1, :M-1]
    ym = ym - N_half + 0.5
    xm = xm - M_half + 0.5 
    
    # Calculate the diagonal gradients of intensities across each
    # corner of 4 pixels
    dI_du = psf_image[:N-1, 1:] - psf_image[1:, :M-1]
    dI_dv = psf_image[:N-1, :M-1] - psf_image[1:, 1:]
    
    # Smooth the image to reduce the effect of noise, at the cost
    # of a little resolution
    fdu = ndi.uniform_filter(dI_du, 3)
    fdv = ndi.uniform_filter(dI_dv, 3)
    
    dI2 = (fdu ** 2) + (fdv ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = -(fdv + fdu) / (fdu - fdv)
        
    # For pixel values that blow up, instead set them to a very
    # high float
    m[np.isinf(m)] = 9e9
    
    b = ym - m * xm

    sdI2 = dI2.sum()
    ycentroid = (dI2 * ym).sum() / sdI2
    xcentroid = (dI2 * xm).sum() / sdI2
    w = dI2 / np.sqrt((xm - xcentroid)**2 + (ym - ycentroid)**2)

    # Correct nan / inf values
    w[np.isnan(m)] = 0
    b[np.isnan(m)] = 0
    m[np.isnan(m)] = 0

    # Least-squares analytical solution to the point of 
    # maximum radial symmetry, given the slopes at each
    # edge of 4 pixels
    wm2p1 = w / ((m**2) + 1)
    sw = wm2p1.sum()
    smmw = ((m**2) * wm2p1).sum()
    smw = (m * wm2p1).sum()
    smbw = (m * b * wm2p1).sum()
    sbw = (b * wm2p1).sum()
    det = (smw ** 2) - (smmw * sw)
    xc = (smbw*sw - smw*sbw)/det
    yc = (smbw*smw - smmw*sbw)/det

    # Adjust coordinates so that they're relative to the
    # edge of the image frame
    yc = (yc + (N + 1) / 2.0) - 1
    xc = (xc + (M + 1) / 2.0) - 1

    return np.array([yc, xc])

def L_poisson(psf_img, pars, sigma=1.0):
    """
    Compute the model, Jacobian, and Hessian 
    of a 2D integrated Gaussian PSF under a Poisson 
    noise model, given a real image. 

    Used internally in quot.localize.mle_poisson.

    args
    ----
        psf_img : 2D ndarray
        pars : 1D ndarray, [y0, x0, I, bg]
        sigma : float, Gaussian width

    returns
    -------
        (
            2D ndarray of shape psf_img.shape,
                the PSF model;

            2D ndarray of shape (n_pixels, 4),
                the Jacobian;

            2D ndarray of shape (4, 4), the 
                Hessian
        )

    """
    y0, x0, I, bg = pars 

    # Number of pixels
    w = psf_img.shape[0]
    M = w**2

    # Pixel indices
    Y, X = np.indices((w, w))

    # Precompute some factors
    A = 2 * (sigma**2)
    B = np.sqrt(A)
    C = np.pi * A 
    D = np.sqrt(C)

    # Compute PSF 1D projections along Y and 
    # X axes
    E_y = 0.5*(erf((Y-y0+0.5)/B)-erf((Y-y0-0.5)/B))
    E_x = 0.5*(erf((X-x0+0.5)/B)-erf((X-x0-0.5)/B))

    # Compute derivatives of these projections
    # with respect to their axis
    dEy_dy = (np.exp(-((Y-y0-0.5)**2 / A)) - \
        np.exp(-((Y-y0+0.5)**2 / A))) / D 
    dEx_dx = (np.exp(-((X-x0-0.5)**2 / A)) - \
        np.exp(-((X-x0+0.5)**2 / A))) / D    

    # Compute the model PSF
    model = I*E_y*E_x + bg 

    # Only evaluate where the model is nonzero
    nonzero = model > 0
    nonzero_r = nonzero.ravel()

    # Common factor for Jacobian calculations
    J_factor = (psf_img[nonzero]/model[nonzero])-1.0

    # Common factor for Hessian calculations
    H_factor = (psf_img[nonzero]/(model[nonzero]**2))

    # Compute model derivatives 
    du_dj = [
        I*E_x[nonzero]*dEy_dy[nonzero],
        I*E_y[nonzero]*dEx_dx[nonzero],
        E_y[nonzero]*E_x[nonzero],
        np.ones(nonzero.sum())
    ]

    # Compute the Jacobian
    J = np.zeros((M, 4), dtype='float64')
    for j in range(4):
        J[:,j][nonzero_r] = J_factor*du_dj[j]

    # Compute the Hessian 
    H = np.zeros((4, 4), dtype='float64')
    for i in range(4):
        for j in range(i, 4):
            H[i,j] = -(H_factor * \
                du_dj[i] * \
                du_dj[j]).sum()

            # Compute Hessian by symmetry
            if i != j:
                H[j,i] = H[i,j]

    return model, J, H 

def J_int_gaussian(psf_img, pars, sigma=1.0):
    """
    Evaluate the model function and Jacobian
    of a 2D integrated Gaussian PSF model
    under a Gaussian noise model, appropriate
    for nonlinear LS optimizers.

    This model uses the following parameter vector:

        pars[0] : y center
        pars[1] : x center
        pars[2] : Gaussian intensity
        pars[3] : BG intensity / pixel

    The sigma of the Gaussian is assumed constant.

    args
    ----
        psf_img : 2D ndarray
        pars : 1D ndarray, parameter vector
        sigma : float

    returns
    -------
        (
            2D ndarray, the evaluated PSF model;
            2D ndarray of shape (n_pixels, 4), the
                Jacobian
        )

    """
    # Number of pixels
    M = psf_img.shape[0] * psf_img.shape[1]

    # Pixel indices
    Y, X = np.indices(psf_img.shape)

    # Normalization
    var2 = 2 * (sigma**2)

    # Evaluate the intensities of a unit-intensity
    # Gaussian projected onto each axis 
    E_y = psf_int_1d(Y, pars[0], sigma=sigma)
    E_x = psf_int_1d(X, pars[1], sigma=sigma)

    # Evaluate the model function
    model = pars[2]*E_y*E_x + pars[3]

    # Evaluate residuals with respect to model function
    r = (model - psf_img).ravel()

    # Evaluate the derivatives of the projected 
    # Gaussian with respect to their axes
    dEy_dy = (np.exp(-((Y-pars[0]-0.5)**2)/var2) - \
        np.exp(-((Y-pars[0]+0.5)**2)/var2))/(np.pi*var2)
    dEx_dx = (np.exp(-((X-pars[1]-0.5)**2)/var2) - \
        np.exp(-((X-pars[1]+0.5)**2)/var2))/(np.pi*var2)   

    # Evaluate the Jacobian
    J = np.empty((M, 4), dtype='float64')
    J[:,0] = (pars[2] * E_x * dEy_dy).ravel()
    J[:,1] = (pars[2] * E_y * dEx_dx).ravel()
    J[:,2] = (E_y * E_x).ravel()
    J[:,3] = np.ones(M, dtype='float64')

    return model, J 

def J_point_gaussian(psf_img, pars, sigma=1.0):
    """
    Evaluate the model function and Jacobian
    of a 2D pointwise Gaussian PSF model
    under a Gaussian noise model, appropriate
    for nonlinear LS optimizers.

    This model uses the following parameter vector:

        pars[0] : y center
        pars[1] : x center
        pars[2] : Gaussian intensity
        pars[3] : BG intensity / pixel

    The sigma of the Gaussian is assumed constant.

    args
    ----
        psf_img : 2D ndarray
        pars : 1D ndarray, parameter vector
        sigma : float

    returns
    -------
        (
            2D ndarray, the evaluated PSF model;
            2D ndarray of shape (n_pixels, 4), the
                Jacobian
        )

    """
    # Number of pixels
    M = psf_img.shape[0] * psf_img.shape[1]

    # Pixel indices
    Y, X = np.indices(psf_img.shape)

    # Normalization 
    var2 = 2 * (sigma**2)

    # Evaluate a unit-intensity Gaussian
    U = np.exp(-((Y-pars[0])**2 + (X-pars[1])**2)/var2) / (var2*np.pi)

    # Evaluate the model
    model = pars[2]*U + pars[3]

    # Evaluate the Jacobian
    J = np.empty((M, 4), dtype='float64')
    J[:,0] = ((Y-pars[0])*pars[2]*U/(sigma**2)).ravel()
    J[:,1] = ((X-pars[1])*pars[2]*U/(sigma**2)).ravel()
    J[:,2] = U.ravel()
    J[:,3] = np.ones(M, dtype='float64')

    return model, J 

#
# MISCELLANEOUS UTILITIES
#
def prune_kwargs(function, **kwargs):
    """
    For one of the detection functions, remove 
    kwargs that are not taken by that function.

    This is not currently used - if the config
    files are written correctly, it should not
    be necessary.

    args
    ----
        function : one of the detection 
            functions

        kwargs : proposed kwargs to function

    returns
    -------
        dict, revised kwargs

    """
    # Get the list of all arguments
    arglist = inspect.getargspec(function).args 

    # Remove the 'img' and 'return_filt' arguments
    arglist = utils.try_list_remove(arglist, \
        'img', 'return_filt')

    # Return the modified dictionary
    return {i: kwargs[i] for i in kwargs.keys() \
        if i in arglist}

#
# PLOTTING UTILITIES
#
def wireframe_overlay(*imgs):
    """
    Overlay 2D distributions using a wireframe
    visualization.

    args
    ----
        *imgs : 2D ndarrays

    """
    import matplotlib.pyplot as plt 
    from mpl_toolkits.mplot3d import Axes3D 
    colors = ['k', 'r', 'b']

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    for j, img in enumerate(imgs):
        Y, X = np.indices(img.shape)
        if j < 3:
            ax.plot_wireframe(
                X, Y, img, color=colors[j]
            )
        else:
            ax.plot_wireframe(X, Y, img)
        ax.set_zlim((0, ax.get_zlim()[1]))
    plt.show(); plt.close()


# 
# LOCALIZATION DATAFRAME UTILITIES
#
def attrib_histogram_2d(locs, attrib_0, attrib_1, n_bins=150):
    """
    Make a 2D histogram of the density of each 
    attribute, for potential masking.

    args
    ----
        locs : pandas.DataFrame
        attrib_0, attrib_1: str, columns in locs 

    returns
    -------
        (
            2D ndarray, the density,
            1D ndarray, the bin edges for attrib_0
            1D ndarray, the bin edges for attrib_1
        )

    """
    # Make sure the dataframe contains the desired columns
    assert attrib_0 in locs.columns
    assert attrib_1 in locs.columns

    # Format as ndarray for speed
    X = np.asarray(locs[[attrib_0, attrib_1]])

    # Exclude outliers
    y0 = max([
        X[:,0].mean() - 3*X[:,0].std(),
        X[:,0].min()
    ])
    y1 = min([
        X[:,0].mean() + 3*X[:,0].std(),
        X[:,0].max()
    ])
    x0 = max([
        X[:,1].mean() - 3*X[:,1].std(),
        X[:,1].min()
    ])
    x1 = min([
        X[:,1].mean() + 3*X[:,1].std(),
        X[:,1].max()
    ])

    # Make the bins in each axis
    bin_edges_0 = np.linspace(y0, y1, n_bins+1)
    bin_edges_1 = np.linspace(x0, x1, n_bins+1)

    # Make the 2D histogram
    H, _edges_0, _edges_1 = np.histogram2d(
        X[:,0], X[:,1], bins=(bin_edges_0, bin_edges_1))

    return H, bin_edges_0, bin_edges_1 

#
# TRACKING UTILITIES
#
def connected_components(semigraph):
    '''
    Find independent subgraphs in a semigraph by a floodfill procedure.
    
    args
        semigraph : 2D binary np.array (only 0/1 values), representing
            a semigraph
    
    returns
        subgraphs : list of 2D np.array, the independent adjacency subgraphs;
        
        subgraph_y_indices : list of 1D np.array, the y-indices of each 
            independent adjacency subgraph;
            
        subgraph_x_indices : list of 1D np.array, the x-indices of each 
            independent adjacency subgraph;
        
        y_without_x : 1D np.array, the y-indices of y-nodes without any edges
            to x-nodes;
        
        x_without_y : 1D np.array, the x-indices of x-nodes without any edges
            to y-nodes.
            
    '''
    if semigraph.max() > 1:
        raise RuntimeError("connected_components only takes binary arrays")
        
    # The set of all y-nodes (corresponding to y-indices in the semigraph)
    y_indices = np.arange(semigraph.shape[0]).astype('uint16')
    
    # The set of all x-nodes (corresponding to x-indices in the semigraph)
    x_indices = np.arange(semigraph.shape[1]).astype('uint16')

    # Find y-nodes that don't connect to any x-node,
    # and vice versa
    where_y_without_x = (semigraph.sum(axis = 1) == 0)
    where_x_without_y = (semigraph.sum(axis = 0) == 0)
    y_without_x = y_indices[where_y_without_x]
    x_without_y = x_indices[where_x_without_y]
    
    # Consider the remaining nodes, which have at least one edge
    # to a node of the other class 
    semigraph = semigraph[~where_y_without_x, :]
    semigraph = semigraph[:, ~where_x_without_y]
    y_indices = y_indices[~where_y_without_x]
    x_indices = x_indices[~where_x_without_y]
    
    # For the remaining nodes, keep track of (1) the subgraphs
    # encoding connected components, (2) the set of original y-indices
    # corresponding to each subgraph, and (3) the set of original x-
    # indices corresponding to each subgraph
    subgraphs = []
    subgraph_y_indices = []
    subgraph_x_indices = []

    # Work by iteratively removing independent subgraphs from the 
    # graph. The list of nodes still remaining are kept in 
    # *unassigned_y* and *unassigned_x*
    unassigned_y, unassigned_x = (semigraph == 1).nonzero()
    
    # The current index is used to floodfill the graph with that
    # integer. It is incremented as we find more independent subgraphs. 
    current_idx = 2
    
    # While we still have unassigned nodes
    while len(unassigned_y) > 0:
        
        # Start the floodfill somewhere with an unassigned y-node
        semigraph[unassigned_y[0], unassigned_x[0]] = current_idx
    
        # Keep going until subsequent steps of the floodfill don't
        # pick up additional nodes
        prev_nodes = 0
        curr_nodes = 1
        while curr_nodes != prev_nodes:
            # Only floodfill along existing edges in the graph
            where_y, where_x = (semigraph == current_idx).nonzero()
            
            # Assign connected nodes to the same subgraph index
            semigraph[where_y, :] *= current_idx
            semigraph[semigraph > current_idx] = current_idx
            semigraph[:, where_x] *= current_idx
            semigraph[semigraph > current_idx] = current_idx
            
            # Correct for re-finding the same nodes and multiplying
            # them more than once (implemented in the line above)
            # semigraph[semigraph > current_idx] = current_idx
            
            # Update the node counts in this subgraph
            prev_nodes = curr_nodes
            curr_nodes = (semigraph == current_idx).sum()
        current_idx += 1 

        # Get the local indices of the y-nodes and x-nodes (in the context
        # of the remaining graph)
        where_y = np.unique(where_y)
        where_x = np.unique(where_x)

        # Use the local indices to pull this subgraph out of the 
        # main graph 
        subgraph = semigraph[where_y, :]
        subgraph = subgraph[:, where_x]

        # Save the subgraph
        if not (subgraph.shape[0] == 0 and subgraph.shape[0] == 0):
            subgraphs.append(subgraph)
        
            # Get the original y-nodes and x-nodes that were used in this
            # subgraph
            subgraph_y_indices.append(y_indices[where_y])
            subgraph_x_indices.append(x_indices[where_x])

        # Update the list of unassigned y- and x-nodes
        unassigned_y, unassigned_x = (semigraph == 1).nonzero()

    return subgraphs, subgraph_y_indices, subgraph_x_indices, y_without_x, x_without_y

def sq_radial_distance(vector, points):
    return ((vector - points) ** 2).sum(axis = 1)

def sq_radial_distance_array(points_0, points_1):
    '''
    args
        points_0    :   np.array of shape (N, 2), coordinates
        points_1    :   np.array of shape (M, 2), coordinates

    returns
        np.array of shape (N, M), the radial distances between
            each pair of points in the inputs

    '''
    array_points_0 = np.zeros((points_0.shape[0], points_1.shape[0], 2), dtype = 'float')
    array_points_1 = np.zeros((points_0.shape[0], points_1.shape[0], 2), dtype = 'float')
    for idx_0 in range(points_0.shape[0]):
        array_points_0[idx_0, :, :] = points_1 
    for idx_1 in range(points_1.shape[0]):
        array_points_1[:, idx_1, :] = points_0
    result = ((array_points_0 - array_points_1)**2).sum(axis = 2)
    return result 

