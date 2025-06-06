import trackpy as tp
import numpy as np
import pandas as pd

# A new version of tp.motion.emsd() that calculates standard deviation.
# This function is copied from trackpy. (Please see the trackpy license.)
# I [Viva] added the calculation of biased weighted standard deviation.

def stdev_emsd(traj, mpp, fps, max_lagtime=100, detail=False, pos_columns=None):
    """Compute the ensemble mean squared displacements of many particles.

    Parameters
    ----------
    traj : DataFrame of trajectories of multiple particles, including
        columns particle, frame, x, and y
    mpp : microns per pixel
    fps : frames per second
    max_lagtime : intervals of frames out to which MSD is computed
        Default: 100
    detail : Set to True to include <x>, <y>, <x^2>, <y^2>. Returns
        only <r^2> by default.

    Returns
    -------
    Series[msd, index=t] or, if detail=True,
    DataFrame([<x>, <y>, <x^2>, <y^2>, msd, N, lagt,
               std_<x>, std_<y>, std_<x^2>, std_<y^2>, 
               std_msd],
              index=frame)

    Notes
    -----
    Input units are pixels and frames. Output units are microns and seconds.
    """
    ids = []
    msds = []
    for pid, ptraj in traj.reset_index(drop=True).groupby('particle'):
        msds.append(tp.motion.msd(ptraj, mpp, fps, max_lagtime, True, pos_columns))
        ids.append(pid)
    msds = tp.utils.pandas_concat(msds, keys=ids, names=['particle', 'frame'])
    results = msds.mul(msds['N'], axis=0).groupby(level=1).mean()  # weighted average
    results = results.div(msds['N'].groupby(level=1).mean(), axis=0)  # weights normalized
    # Above, lagt is lumped in with the rest for simplicity and speed.
    # Here, rebuild it from the frame index.
    
    if not detail:
        return results.set_index('lagt')['msd']
    
    # correctly compute the effective number of independent measurements
    results['N'] = msds['N'].groupby(level=1).sum()

    # Calculation of biased weighted standard deviation
    numerator = ((msds.subtract(results))**2).mul(msds['N'], axis=0).groupby(level=1).sum()
    denominator = msds['N'].groupby(level=1).sum() - 1 # with Bessel's correction
    variance = numerator.div(denominator, axis=0)
    variance = variance[['<x>', '<y>', '<x^2>','<y^2>','msd']]
    std = np.sqrt(variance)
    std.columns = 'std_' + std.columns  
    #stderr = std/np.sqrt(msds['N'])
    
    return results.join(std)

# usage
#detailed_emsd = stdev_emsd(linked_trajectory_df, scaling, fps, detail=True, max_lagtime=500)


def multi_stdev_emsd(traj_list, mpp_list, fps, max_lagtime=100, detail=False, pos_columns=None):
    """Compute the ensemble mean squared displacements of many particles 
       from a list of trajectory dataframes.

    Parameters
    ----------
    traj_list : List of DataFrames of trajectories of multiple particles,
        with dataframes including columns particle, frame, x, and y
    mpp_list : microns per pixel list, in the same order as traj_list
    fps : frames per second
    max_lagtime : intervals of frames out to which MSD is computed
        Default: 100
    detail : Set to True to include <x>, <y>, <x^2>, <y^2>. Returns
        only <r^2> by default.

    Returns
    -------
    Series[msd, index=t] or, if detail=True,
    DataFrame([<x>, <y>, <x^2>, <y^2>, msd, N, lagt,
               std_<x>, std_<y>, std_<x^2>, std_<y^2>, 
               std_msd],
              index=frame)
    The frame index specifically indicates the difference in frames.

    Notes
    -----
    Input units are pixels and frames. Output units are microns and seconds.
    """
    ids = []
    msds = []
    for traj,mpp in zip(traj_list, mpp_list):
        for pid, ptraj in traj.reset_index(drop=True).groupby('particle'):
            msds.append(tp.motion.msd(ptraj, mpp, fps, max_lagtime, True, pos_columns))
            ids.append(pid)
    msds = tp.utils.pandas_concat(msds, keys=ids, names=['particle', 'frame'])
    results = msds.mul(msds['N'], axis=0).groupby(level=1).mean()  # weighted average
    results = results.div(msds['N'].groupby(level=1).mean(), axis=0)  # weights normalized
    # Above, lagt is lumped in with the rest for simplicity and speed.
    # Here, rebuild it from the frame index.
    
    if not detail:
        return results.set_index('lagt')['msd']
    
    # correctly compute the effective number of independent measurements
    results['N'] = msds['N'].groupby(level=1).sum()

    # Calculation of biased weighted standard deviation
    numerator = ((msds.subtract(results))**2).mul(msds['N'], axis=0).groupby(level=1).sum()
    denominator = msds['N'].groupby(level=1).sum() - 1 # with Bessel's correction
    variance = numerator.div(denominator, axis=0)
    variance = variance[['<x>', '<y>', '<x^2>','<y^2>','msd']]
    std = np.sqrt(variance)
    std.columns = 'std_' + std.columns  
    #stderr = std/np.sqrt(msds['N'])
    
    return results.join(std)

#multi_emsd = multi_stdev_emsd(linked_trajectory_df, scaling, fps, detail=True, max_lagtime=500)

