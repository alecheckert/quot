#!/usr/bin/env python
"""
track.py -- reconnect localizations into trajectories

"""
# Numeric
import numpy as np 

# Dataframes
import pandas as pd 

# Distance between two sets of 2D points
from scipy.spatial import distance_matrix 

# Hungarian algorithm
from munkres import Munkres 
hungarian_solver = Munkres()

# Custom utilities
from .helper import (
    connected_components
)

##################################
## LOW-LEVEL TRACKING UTILITIES ##
##################################

class Trajectory():
    """
    Convenience class used internally by track_locs().

    A Trajectory object specifies a set of indices to
    localizations in a large array that are to be reconnected
    into trajectories.

    It also holds a reference to the original array, so 
    that it can grab information about its localization when 
    necessary.

    When Trajectories are not reconnected in a given frame,
    their blink counter (self.n_blinks) is incremented. When
    this exceeds *max_blinks*, the Trajectories are marked
    for termination.

    The Trajectory class is also convenient to hold associated
    information about the tracking problem, such as the 
    number of competing trajectories and localizations, etc.,
    that are returned at the end of tracking.

    init
    ----
        start_idx       :   int, the index of the first 
                            localization in this Trajectory
        locs            :   2D ndarray, all localizations
        subproblem_shape:   (int, int), the number of trajs
                            and locs in the subproblem that 
                            created this trajectory
        max_blinks      :   int, the maximum tolerated number
                            of gaps in tracking

    """
    def __init__(self, start_idx, locs, subproblem_shape, max_blinks=0):
        self.indices = [int(start_idx)]
        self.locs = locs 
        self.max_blinks = max_blinks
        self.n_blinks = 0
        self.active = True 
        self.subproblem_shapes = [subproblem_shape]

    def add_index(self, idx, subproblem_shape):
        """
        Extend this trajectory by one localization.

        args
        ----
            idx     :   int, the index of the localization 
                        in self.locs to add
            subproblem_shape    :   (int, int), the size 
                        of the tracking subproblem in which
                        this localization was added

        """
        self.indices.append(int(idx))
        self.subproblem_shapes.append(subproblem_shape)

    def blink(self):
        """
        Skip a frame. If a Trajectory has been in blink for
        more than *self.n_blinks* frames, it marks itself
        for termination by setting self.active = False.

        """
        self.n_blinks += 1
        if self.n_blinks > self.max_blinks:
            self.active = False 

    def get_slice(self):
        """
        Return the slice of the localization array that 
        corresponds to this trajectory.

        returns
        -------
            2D ndarray of shape (traj_len, 5)

        """
        return self.locs[tuple(self.indices), :]

    def last_pos(self):
        """
        Return the last known position of this Trajectory.

        returns
        -------
            (float y, float x), in pixels

        """
        return self.locs[self.indices[-1], 2:4]

def traj_loc_distance(trajs, locs):
    """
    Return the distance between each trajectory and each 
    localization.

    args
    ----
        trajs       :   list of Trajectory
        locs        :   2D ndarray with columns loc_idx,
                        frame, y, x, I0

    returns
    -------
        2D ndarray D, where D[i,j] is the distance between
            Trajectory i and localization j

    """
    return distance_matrix(
        np.asarray([t.last_pos() for t in trajs]),
        locs[:,2:4]
    )

def diffusion_weight_matrix(trajs, locs, frame_interval=0.00548,
    pixel_size_um=0.16, k_return_from_blink=1.0, d_max=5.0, 
    y_diff=0.9, search_radius=2.5, d_bound_naive=0.1, init_cost=50.0):
    """
    Generate the weight matrix for reconnection between a set of 
    Trajectories and a set of localizations for the "diffusion"
    method.

    In this method, the weight of reconnecting trajectory A to
    localization B is equal to the negative log likelihood of 
    trajectory A diffusing to localization B in the relevant
    frame interval (equal to *frame_interval* if there are no 
    gaps). The likelihood is evaluated with a 2D Brownian motion
    model.

    A weighted combination of two such negative log-likelihoods
    is used. The first assumes a diffusion coefficient equal to 
    the maximum likelihood diffusion coefficient for that 
    trajectory (using the MSD method). The second assumes a 
    diffusion coefficient equal to d_max. The relative weights
    of the two estimates are set by *y_diff*.

    args
    ----
        trajs           :   list of Trajectory
        locs            :   2D ndarray, localizations to consider
                            for connection
        frame_interval  :   float, seconds
        pixel_size_um   :   float, um
        k_return_from_blink :   float, penalty to return a trajectory
                                from blinking status
        d_max           :   float, the maximum expected diffusion
                            coefficient in um^2 s^-1
        y_diff          :   float, the relative influence of the 
                            particle's local history on its estimated
                            diffusion coefficient
        search_radius   :   float, um
        d_bound_naive   :   float, naive estimate for a particle's
                            local diffusion coefficient, um^2 s^-1
        init_cost       :   float, static cost to initialize a new
                            trajectory when reconnections are available
                            in the search radius

    returns
    -------
        2D ndarray of shape (n_trajs, n_locs), the weights
            for reconnection

    """
    n_traj = len(trajs)
    n_locs = locs.shape[0]
    n_dim = n_traj + n_locs

    # Unit conversions
    search_radius_pxl = search_radius / pixel_size_um
    max_var2 = 2.0 * d_max * frame_interval / (pixel_size_um ** 2)

    # The weight matrix
    W = np.zeros((n_dim, n_dim), dtype="float64")

    # Set the static cost of starting a new trajectory
    for li in range(n_locs):
        W[n_traj:, li] = init_cost
    W[:, n_locs:] = init_cost

    # For each trajectory, calculating the weight of
    # assignments to each localization
    for ti in range(n_traj):

        # Last known position of this trajectory
        last_pos = trajs[ti].last_pos()

        # Penalize blinking trajectories
        L_blink = -k_return_from_blink * trajs[ti].n_blinks + np.log(k_return_from_blink)

        # Distances to each localization
        R = traj_loc_distance([trajs[ti]], locs)[0, :]
        R2 = R ** 2

        # Estimate the local diffusion coefficient of this
        # trajectory. 

        # If no prior displacements are available, use the 
        # naive estimate
        if len(trajs[ti].indices) == 1:
            local_var2 = 2*d_bound_naive*frame_interval * \
                (1+trajs[ti].n_blinks)

        # Otherwise, use the MSD method
        else:
            traj_slice = trajs[ti].get_slice()
            frames = traj_slice[:,1].astype('int64')
            pos = traj_slice[:,2:4]
            delta_frames = frames[1:] - frames[:-1]
            local_var2 = 0.5 * (((pos[1:,:]-pos[:-1,:])**2).sum(1) / \
                delta_frames).mean() * (1+trajs[ti].n_blinks)

        # Log-likelihood of diffusing from last
        # known position to each new position
        L_diff = np.log(
            y_diff * (R/local_var2) * np.exp(-R2/(2*local_var2))
            + (1-y_diff) * (R/max_var2) * np.exp(-R2/(2*max_var2))
        )

        # Make sure we do NOT reconnect trajectories to localizations
        # outside of their search radii
        L_diff[R>search_radius_pxl] = -np.inf

        # Assign reconnection weight
        W[ti,:n_locs] = -(L_blink+L_diff)

    return W

def euclidean_weight_matrix(trajs, locs, pixel_size_um=0.16, 
    scale=1.0, search_radius=2.5, init_cost=50.0, **kwargs):
    """
    Generate the weight matrix for reconnection between 
    Trajectories and localizations for the "euclidean"
    reconnection method.

    Here, the weight to reconnect traj I with localization J
    is just the distance between the last known position of I
    and J, scaled by the constant *scale*.

    If J is outside the search radius of I, the weight is 
    infinite.

    The weight to drop *I* or to start a new trajectory from
    *J* when other reconnections are available is *init_cost*.

    args
    ----
        trajs           :   list of Trajectory
        locs            :   2D ndarray, localizations to consider
                            for connection
        pixel_size_um   :   float, um
        scale           :   float, inflation factor for the distances
        search_radius   :   float, um
        init_cost       :   float, penalty for not performing   
                            available reconnections
        kwargs          :   discarded

    returns
    -------
        2D ndarray of shape (n_trajs, n_locs), the reconnection
            weights

    """
    n_traj = len(trajs)
    n_locs = locs.shape[0]
    n_dim = n_traj + n_locs 

    # Weight matrix
    W = np.zeros((n_dim, n_dim), dtype='float64')

    # Distance from each trajectory to each localization
    distances = traj_loc_distance(trajs, locs)

    # Reconnections out of the search radius are impossible
    out_of_radius = distances > search_radius/pixel_size_um 
    distances[out_of_radius] = np.inf 

    # Rescale if desired
    if scale != 1.0:
        distances[~out_of_radius] = distances[~out_of_radius] * scale 

    # Weight of traj:loc reconnection is proportional to Euclidean distance
    W[:n_traj, :n_locs] = distances 

    # Static cost to start new trajectory
    W[n_traj:, :] = init_cost 

    # Penalize reconnecting to nothing
    W[:n_traj, n_locs:] = init_cost 

    return W 

######################################
## FRAME-FRAME RECONNECTION METHODS ##
######################################

def reconnect_conservative(trajs, locs, locs_array, max_blinks=0,
    frame_interval=0.00548, pixel_size_um=0.16):
    """
    Only reassign trajs to locs when the assignment is 
    unambiguous (1 traj, 1 loc within the search radius).

    For all other trajectories, terminate them.
    For all other locs, start new trajectories.

    args
    ----
        trajs           :   list of Trajectory
        locs            :   2D ndarray with columns loc_idx, frame,
                            y, x, I0
        max_blinks      :   int
        frame_interval  :   float
        pixel_size_um   :   float

    returns
    -------
        list of Trajectory

    """
    out = []

    # A single localization and trajectory pair - assignment
    # is unambiguous. This is the only situation where reconnection
    # is allowed.
    n_trajs = len(trajs)
    n_locs = locs.shape[0]
    if n_trajs==1 and n_locs==1:
        trajs[0].add_index(locs[0,0], (n_trajs,n_locs))
        out = trajs 

    # Multiple localizations and/or trajectories
    else:

        # Terminate all existing trajectories
        for ti in range(n_trajs):
            trajs[ti].active = False 
        out += trajs 

        # Start new trajectories from all localizatoins
        for li in range(n_locs):
            out.append(Trajectory(locs[li,0], locs_array, 
                (n_trajs,n_locs), max_blinks=max_blinks))

    return out 

WEIGHT_MATRIX_METHODS = {
    'diffusion': diffusion_weight_matrix,
    'euclidean': euclidean_weight_matrix   
}

def reconnect_hungarian(trajs, locs, locs_array, max_blinks=0,
    weight_method=None, min_I0=0.0, **kwargs):
    """
    Assign Trajectories to localizations by assigning each 
    possible reconnection a weight, then finding the assignment
    that minimizes the summed weights with the Hungarian 
    algorithm.

    args
    ----
        trajs           :   list of Trajectory
        locs            :   2D ndarray, localizations to consider
                            for connection
        locs_array      :   2D ndarray, all localizations in this
                            movie
        max_blinks      :   int
        weight_method   :   str, the method to use to generate
                            the weight matrix
        min_I0          :   float, minimum intensity to start 
                            a new Trajectory
        kwargs          :   to weight_method

    returns
    -------
        list of Trajectory

    """
    out = []

    # Get the size of the assignment problem
    n_traj = len(trajs)
    n_locs = locs.shape[0]

    # Unambiguous - only one trajectory and localization
    if n_traj==1 and n_locs==1:
        trajs[0].add_index(locs[0,0], (n_traj,n_locs))

    # Otherwise, solve the assignment problem by finding 
    # weights and minimizing with the Hungarian algorithm
    else:

        # Get the weight matrix for reconnection
        W = WEIGHT_MATRIX_METHODS[weight_method](
            trajs, locs, **kwargs)

        # Minimize negative log likelihood with
        # Hungarian algorithm
        for i, j in hungarian_solver.compute(W):

            # traj:loc
            if (i<n_traj) and (j<n_locs):
                trajs[i].add_index(locs[j,0], (n_traj,n_locs))

            # traj:(empty)
            elif (i<n_traj) and (j>=n_locs):
                trajs[i].blink()

            # (empty):loc
            elif (j<n_locs) and (i>=n_traj):
                if locs[j,4] >= min_I0:
                    out.append(Trajectory(locs[j,0], locs_array,
                        (n_traj,n_locs)))
            else:
                pass

    # Combine new and existing trajs
    out += trajs
    return out

def reconnect_diffusion(trajs, locs, locs_array, max_blinks=0, 
    min_I0=0.0, **kwargs):
    """
    Assign Trajectories to localizations on the basis of their
    expected probability of diffusion and their blinking status.

    Each of the Trajectories is assumed to be a Brownian motion
    in 2D. Its diffusion coefficient is evaluated from its 
    history by MSD if it is greater than length 1, or from 
    d_bound_naive otherwise.

    args
    ----
        trajs           :   list of Trajectory
        locs            :   2D ndarray, localizations to consider
                            for connection
        locs_array      :   2D ndarray, all localizations in this
                            movie
        max_blinks      :   int
        frame_interval  :   float, seconds
        pixel_size_um   :   float, um
        min_I0          :   float, AU
        k_return_from_blink :   float, penalty to return a trajectory
                                from blinking status
        d_max           :   float, the maximum expected diffusion
                            coefficient in um^2 s^-1
        y_diff          :   float, the relative influence of the 
                            particle's local history on its estimated
                            diffusion coefficient
        search_radius   :   float, um
        d_bound_naive   :   float, naive estimate for a particle's
                            local diffusion coefficient, um^2 s^-1

    returns
    -------
        list of Trajectory

    """
    return reconnect_hungarian(trajs, locs, locs_array, 
        weight_method='diffusion', max_blinks=0, min_I0=min_I0, **kwargs)

def reconnect_euclidean(trajs, locs, locs_array, max_blinks=0,
    min_I0=0.0, **kwargs):
    """
    Assign Trajectories to localizations purely by minimizing
    the total Trajectory-localization distances. 

    args
    ----
        trajs           :   list of Trajectory
        locs            :   2D ndarray, localizations to consider
                            for connection
        locs_array      :   2D ndarray, all localizations in this
                            movie
        max_blinks      :   int
        min_I0          :   float, minimum intensity to start a
                            new trajectory
        pixel_size_um   :   float, um
        scale           :   float, inflation factor for the distances
        search_radius   :   float, um
        init_cost       :   float, cost to start a new trajectory
                            if reconnections are available

    returns
    -------
        list of Trajectory

    """
    return reconnect_hungarian(trajs, locs, locs_array, 
        weight_method='euclidean', max_blinks=0, min_I0=min_I0, **kwargs)

########################################
## ALL AVAILABLE RECONNECTION METHODS ##
########################################

METHODS = {
    'conservative': reconnect_conservative,
    'diffusion': reconnect_diffusion,
    'euclidean': reconnect_euclidean
}

############################
## MAIN TRACKING FUNCTION ##
############################

def track(locs, method="diffusion", search_radius=2.5, 
    pixel_size_um=0.16, frame_interval=0.00548, min_I0=0.0,
    max_blinks=0, debug=False, **kwargs):
    """
    Given a dataframe with localizations, reconnect into 
    trajectories.

    Each frame-frame reconnection problem is considered 
    separately and sequentially. For each problem:

        1. Figure out which localizations lie within the
            the search radii of the current trajectories
        2. Identify disconnected "subproblems" in this 
            trajectory-localization adjacency map
        3. Solve all of the subproblems by a method 
            specified by the *method* kwarg
        4. Update the trajectories and proceed to the 
            next frame

    The result is an assignment of each localization to a
    trajectory index. Localizations that were not reconnected
    into a trajectory for whatever reason are assigned a 
    trajectory index of -1. 

    args
    ----
        locs            :   pandas.DataFrame, set of localizations

    returns
    -------
        pandas.Series, trajectory indices for each localization.

    """
    start_frame = int(locs['frame'].min())
    stop_frame = int(locs['frame'].max())+1

    # Get the reconnection method
    method_f = METHODS.get(method)

    # Sort the localizations by frame (unecessary, but easier
    # to interpret)
    locs = locs.sort_values(by='frame')

    # Assign each localization a unique index
    locs['loc_idx'] = np.arange(len(locs))

    # Convert locs to ndarray for speed
    cols = ['loc_idx', 'frame', 'y', 'x', 'I0']
    L = np.asarray(locs[cols])

    # Maximum tolerated traj-loc jump distance (search radius)
    search_radius_pxl = search_radius / pixel_size_um 

    # Convenience function: get all of locs from one frame
    def get_locs(frame):
        return L[L[:,1]==frame,:]

    # Convenience function: in a list of Trajectory, find 
    # trajectories that have finished
    def get_finished(trajs):
        _finished = [t for t in filter(lambda i: ~i.active, trajs)]
        _active = [t for t in filter(lambda i: i.active, trajs)]
        return _active, _finished

    # Start by grabbing the locs in the first frame and 
    # initializing Trajectories from each of them 
    frame_locs = get_locs(start_frame)
    active = [Trajectory(int(i), L, (0,1)) for i in frame_locs[:,0]]

    # During tracking, Trajectories are tossed between 
    # three categories: "active", "new", and "completed". 
    # "active" Trajectories are eligible for reconnection in 
    # this frame, "new" Trajectories will become active 
    # Trajectories in the next frame, and "completed" Trajectories
    # have been removed from the pool.
    new = []
    completed = []

    for fi in range(start_frame+1, stop_frame):
        frame_locs = get_locs(fi)

        # If there are no locs in this frame, set all active
        # trajectories into blink
        if len(frame_locs.shape)<2 or frame_locs.shape[0]==0:

            # Increment blink counter
            for t in active: t.blink()

            # Find which trajectories are finished
            active, done = get_finished(active)
            completed += done 

            # To next frame
            continue 

        # If there are no active trajectories, consider starting
        # one from each localization if it passes the intensity 
        # threshold
        elif len(active)==0:

            for i in frame_locs[frame_locs[:,4]>=min_I0, 0]:
                new.append(Trajectory(int(i), L, (0,1)))
            active = new 
            new = []

            # To next frame
            continue 

        # Otherwise, there is some combination of active trajectories
        # and localizations in this frame.
        else:

            # Calculate the adjacency graph: which localizations are 
            # within the search radius of which trajectories?
            adj_g = (traj_loc_distance(active, frame_locs) <=
                search_radius_pxl).astype(np.int64)

            # Break this graph into subgraphs, each of which represents
            # a separate tracking subproblem
            subgraphs, Ti, Li, traj_singlets, loc_singlets = \
                connected_components(adj_g)

            # If a trajectory does not have localizations in its 
            # search radius, set it into blink
            for ti in traj_singlets:
                active[ti].blink()
                if active[ti].active:
                    new.append(active[ti])
                else:
                    completed.append(active[ti])

            # If a localization has no nearby trajectories, start
            # a new trajectory if it passes the intensity threshold
            for li in loc_singlets:
                if frame_locs[li,4] >= min_I0:
                    new.append(Trajectory(frame_locs[li,0], L, (0,1)))

            # If there are both trajectories and localizations in the 
            # subproblem, reconnect according to the reconnection method
            for si, subgraph in enumerate(subgraphs):

                # Only one traj and one loc: assignment is unambiguous
                if subgraph.shape[0]==1 and subgraph.shape[1]==1:
                    active[Ti[si][0]].add_index(frame_locs[Li[si][0], 0], (1,1))
                    new.append(active[Ti[si][0]])

                # Otherwise, pass to the reconnection method
                else:
                    out_trajs = method_f([active[i] for i in Ti[si]],
                        frame_locs[Li[si],:], L, max_blinks=max_blinks,
                        pixel_size_um=pixel_size_um, frame_interval=frame_interval,
                        **kwargs)

                    # Find finished trajectories
                    not_done, done = get_finished(out_trajs)
                    completed += done
                    new += not_done 

            # For trajs eligible for reconnection in the next frame,
            # transfer to *active*
            active = new 
            new = []

    # Finish any trajectories still running
    completed += active 

    # Trajectory indices
    ids = np.full(L.shape[0], -1, dtype=np.int64)

    # Number of competing trajectories and competing localizations
    # for the subproblem in which each localization was connected
    # (1: no competition)
    subproblem_sizes_traj = np.full(L.shape[0], -1, dtype=np.int64)
    subproblem_sizes_locs = np.full(L.shape[0], -1, dtype=np.int64)

    # For each trajectory, add its information to these arrays
    for ti, t in enumerate(completed):
        indices = np.asarray(t.indices)
        T_size = [t.subproblem_shapes[i][0] for i in range(len(indices))]
        L_size = [t.subproblem_shapes[i][1] for i in range(len(indices))]
        ids[np.asarray(t.indices)] = ti 
        subproblem_sizes_traj[np.asarray(t.indices)] = T_size 
        subproblem_sizes_locs[np.asarray(t.indices)] = L_size 

    # Assign traj index as a column in the original dataframe
    locs['trajectory'] = ids 
    locs['subproblem_n_traj'] = subproblem_sizes_traj
    locs['subproblem_n_locs'] = subproblem_sizes_locs

    # If desired, return the Trajectory objects for testing
    if debug:
        return locs, completed 
    else:
        return locs 

