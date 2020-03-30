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

# 
# BASIC UTILITIES ON NDARRAYS AND FLOATS
#
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
        return (img-camera_bg)/camera_gain
    else:
        return img 

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
    sigma=1.0):
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
    return stable_divide_float(
        psf_img[ym,xm] - bg,
        psf_int_1d(ym, y0, sigma=sigma) * \
            psf_int_1d(xm, x0, sigma=sigma),
        inf=np.nan, 
    )

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


