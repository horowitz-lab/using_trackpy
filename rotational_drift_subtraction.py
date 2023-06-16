# Rotational drift subtraction starts here. First we need to define a bunch of functions.
import numpy as np
import pandas as pd
import numba
import matplotlib.pyplot as plt

"""
Known issue:
PendingDeprecationWarning: the matrix subclass is not the recommended way to represent matrices 
or deal with linear algebra (see https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). 
Please adjust your code to use regular ndarray.
"""


## concatenate a new numerical column to a matrix
def put_z_position_in_matrix(mat2D, z=0):
    z_position = np.zeros(len(mat2D)) + z
    z_position = np.matrix(z_position)
    
    mat3D = np.concatenate((mat2D.T, z_position))
    mat3D = mat3D.T
    
    return mat3D

## Check to see if dataframe has z column; otherwise assume z=0.
def get_3D_matrix_from_dataframe(df, xlabel='x',ylabel='y',zlabel='z'):
    try:
        matrix = np.mat(df[[xlabel,ylabel,zlabel]])
    except KeyError:
        matrix = np.mat(df[[xlabel,ylabel]])
        matrix = put_z_position_in_matrix(matrix,0)
        
    return matrix

## The variable A_3D will be a matrix consisting of 
## all coordinates in frame i 
## whose particle is also tracked in frame f.

## The variable B_3D will be a matrix consisting of 
## all coordinates in frame i 
## whose particle is also tracked in frame f.

## This function currently assumes the particles tracked in the image frame 
## are all at the same z.

def matrices_from_dataframe(t1, framei, framef=None, z=0):
    
    # set default for final frame
    if framef == None:
        framef = framei+1
    
    # an inner merge will drop any rows for 
    # particles that are not in both frames 
    AB = pd.merge(t1[t1['frame'] == framei], 
                  t1[t1['frame'] == framef], 
                  how='inner', 
                  on='particle',
                  suffixes=('_i','_f'))

    # Pull out the coordinates and convert to matrices.
    # If z positions are not available, they are set to zero.
    A_3D = get_3D_matrix_from_dataframe(AB, xlabel='x_i',ylabel='y_i',zlabel='z_i')
    B_3D = get_3D_matrix_from_dataframe(AB, xlabel='x_f',ylabel='y_f',zlabel='z_f')
    
    assert len(A_3D) == len(B_3D)
    
    return A_3D, B_3D


## Given a matrix B which 
## has experienced rotation R and translation t, 
## undo that transformation.
def rotational_drift_subtraction(B, R, t):
    n = len(B)
    drift_subtracted = R.T * (B.T - np.tile(t,(1,n)))
    drift_subtracted = drift_subtracted.T
    
    return drift_subtracted

## This function is copied from http://nghiaho.com/uploads/code/rigid_transform_3D.py_
# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
       print("Reflection detected")
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    #print t

    return R, t

# Calculate the axis and angle of rotation for a given rotation matrix R
def axis_angle(R):
    h = R[2,1]
    f = R[1,2]
    c = R[0,2]
    g = R[2,0]
    d = R[1,0]
    b = R[0,1]
    
    # axis of rotation
    axis = [h-f, c-g, d-b]
    
    # angle of rotation, in radians
    angle = np.arccos((np.trace(R) - 1)/2)
    
    ## different way to calculate angle
    # axis_length = np.linalg.norm(axis)
    # angle = np.arcsin(axis_length/2) 
    
    return np.mat(axis), angle # in radians

"""
Unused function.
"""
def print_head(matrix, max_printable_length = 10):
    if len(matrix)>max_printable_length:
        print(matrix[0:max_printable_length])
        print("...")
    else:
        print(matrix)

"""
Calculates, plots, and optionally subtracts translational and rotational COM drift of an ensemble of particles.

@param janus - dataframe of janus particle positions over time
@param tracer - dataframe of tracer particle positions over time
@param do_drift_subtraction - determine if you want drift subtraction to be done

@return dataframes of janus and tracer particles post-drift subtraction
"""
def drift_subtract(janus=None, tracer=None, do_drift_subtraction = True, 
                   show_plots = True, colorseparation=False,
                  movie_name = ""):
    if tracer is None:
        particles = janus.copy()
    else:
        # Before combining janus and tracer into particles, we must make them not have conflicting particle numbers.
        particles = tracer.copy()
        if janus is not None:
            particles['particle'] += janus['particle'].max() + 1
            particles = particles.append(janus) # DEPRECATED! Use pandas.concat instead.
    
    if do_drift_subtraction: # initialize
        janus_nodrift = 0
        del janus_nodrift
        tracer_nodrift = 0
        del tracer_nodrift
    prev_frame = None
    R_list = []
    t_list = []
    x_drifts = []
    y_drifts = []
    z_drifts = []
    axis_list = []
    angle_list = []
    frame_list = []
    verbose = False

    labelx = 'x'
    labely = 'y'
    labelz = 'z'
    transformed_str = '_drift_subtracted'
    labelx2 = labelx + transformed_str
    labely2 = labely + transformed_str
    labelz2 = labelz + transformed_str
    labelnote = 'relative_to_frame'
    
    for current_frame in np.sort(particles.frame.unique()):
        if verbose:
            print("Frame ", current_frame)

        if prev_frame is None:
            relative_to = current_frame
            prev_frame = current_frame
            continue;  # skip first frame

        assert prev_frame is not None

        # A is a shorthand for the previous frame.
        # B is a shorthand for the current frame.

        # Get raw coordinates from current frame and previous frame
        A_3D, B_3D = matrices_from_dataframe(particles, prev_frame, current_frame)

        # Figure out the transformation that occured between frames
        ret_R, ret_t = rigid_transform_3D(A_3D, B_3D)

        # Save a copy of the transformation
        R_list.append(ret_R)
        t_list.append(ret_t)
        x_drifts.append(np.array(ret_t)[0][0])
        y_drifts.append(np.array(ret_t)[1][0])
        z_drifts.append(np.array(ret_t)[2][0])
        current_axis,current_angle = axis_angle(ret_R)
        axis_list.append(current_axis)
        angle_list.append(current_angle)
        frame_list.append(current_frame)

        if do_drift_subtraction:
            ## Do the rotational drift subtraction.
            ## I need to do this with all particles in current frame, 
            ## not just the ones that also appear in previous frame.

            if janus is not None:
                B_dataframe_janus = janus[janus['frame'] == current_frame].copy()
                B_janus = get_3D_matrix_from_dataframe(B_dataframe_janus)
            
            if tracer is not None:
                B_dataframe_tracer = tracer[tracer['frame'] == current_frame].copy()
                B_tracer = get_3D_matrix_from_dataframe(B_dataframe_tracer)

            for R,t in zip(reversed(R_list),reversed(t_list)):
                if verbose:
                    print("undoing transformation")
                    print(R)
                # We use the same R and t to drift subtract both types of particles, 
                # assuming both are in the same overall drifty current.
                if janus is not None:
                    B_janus = rotational_drift_subtraction(B_janus, R, t)
                if tracer is not None:
                    B_tracer = rotational_drift_subtraction(B_tracer, R, t)
                # This is rather brute force, 
                # but I wanted to make sure I'm correct first.
                # The better thing to do is probably to calculate 
                # the total transformation before transforming the coordinates.

            ## Record the drift-subtracted coordinates
            # (i.e. Put the transformed data in the dataframe)

            if janus is not None:
                x_sub_data_janus = np.array(B_janus[:,0]).T[0]
                y_sub_data_janus = np.array(B_janus[:,1]).T[0]
                z_sub_data_janus = np.array(B_janus[:,2]).T[0]
            if tracer is not None:
                x_sub_data_tracer = np.array(B_tracer[:,0]).T[0]
                y_sub_data_tracer = np.array(B_tracer[:,1]).T[0]
                z_sub_data_tracer = np.array(B_tracer[:,2]).T[0]
            if janus is not None:
                B_dataframe_janus[labelx2]=x_sub_data_janus
                B_dataframe_janus[labely2]=y_sub_data_janus
            if tracer is not None:
                B_dataframe_tracer[labelx2]=x_sub_data_tracer
                B_dataframe_tracer[labely2]=y_sub_data_tracer

            # Assumes janus and tracer either both have z data or both don't
            if False: #not np.array_equal(z_sub_data_janus, np.zeros_like(z_sub_data_janus)):
                ## Not tested with a z column
                if janus is not None:
                    B_dataframe_janus[labelz2]=z_sub_data_janus
                if tracer is not None:
                    B_dataframe_tracer[labelz2]=z_sub_data_tracer
                num_new_cols = 4
            else:
                ## no z data
                num_new_cols = 3

            if janus is not None:
                B_dataframe_janus[labelnote] = relative_to
            if tracer is not None:
                B_dataframe_tracer[labelnote] = relative_to

            try:
                if janus is not None:
                    janus_nodrift = pd.concat([janus_nodrift, B_dataframe_janus])
                if tracer is not None:
                    tracer_nodrift = pd.concat([tracer_nodrift, B_dataframe_tracer])
            except NameError:
                # Initialize particles_nodrift
                if janus is not None:
                    janus_nodrift = B_dataframe_janus.copy()
                if tracer is not None:
                    tracer_nodrift = B_dataframe_tracer.copy()

        prev_frame = current_frame

        # end loop

    ## Rename some columns in particles_nodrift
    if do_drift_subtraction:
        # Put the new columns up front
        if colorseparation and janus is not None:
            janus_cols = janus_nodrift.columns.tolist()
            janus_cols = janus_cols[-num_new_cols:]+janus_cols[:-num_new_cols]
            janus_nodrift = janus_nodrift.reindex(columns=janus_cols)
        if tracer is not None:
            tracer_cols = tracer_nodrift.columns.tolist()
            tracer_cols = tracer_cols[-num_new_cols:]+tracer_cols[:-num_new_cols]
            tracer_nodrift = tracer_nodrift.reindex(columns=tracer_cols)

        ## Rename raw columns
        if colorseparation and janus is not None:
            janus_nodrift = janus_nodrift.rename(index=str,
                           columns={labelx: labelx + "_raw", 
                                    labely: labely + "_raw"})
            janus_nodrift = janus_nodrift.rename(index=str,
                           columns={labelx2: labelx,
                                    labely2: labely})
        
        if tracer is not None:
            tracer_nodrift = tracer_nodrift.rename(index=str,
                           columns={labelx: labelx + "_raw", 
                                    labely: labely + "_raw"})
            tracer_nodrift = tracer_nodrift.rename(index=str,
                           columns={labelx2: labelx,
                                    labely2: labely})

        if num_new_cols == 4:
            if tracer is not None:
                ## Not tested with a z column
                tracer_nodrift = tracer_nodrift.rename(index=str, 
                               columns={labelz: labelz + "_raw"})
                tracer_nodrift = tracer_nodrift.rename(index=str,
                               columns={labelz2: labelz})
            
            if colorseparation and janus is not None:
                janus_nodrift = janus_nodrift.rename(index=str, 
                               columns={labelz: labelz + "_raw"})
                janus_nodrift = janus_nodrift.rename(index=str,
                               columns={labelz2: labelz})
            else:
                janus_nodrift= tracer_nodrift
             
    if janus is None:
        janus_nodrift = None
    if tracer is None:
        tracer_nodrift = None
            

    if show_plots:
        # Subplots: https://matplotlib.org/stable/tutorials/introductory/pyplot.html
        plt.subplot(211)        
        plt.plot(frame_list,angle_list)
        plt.title(movie_name + '\nAngular drift\n')
        plt.xlabel('Frame')
        plt.ylabel('Angular drift [radians]')

        plt.subplot(212)
        plt.plot(frame_list,x_drifts, label="x")
        plt.plot(frame_list,y_drifts, label="y")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(movie_name + '\nTranslational drift\n')
        plt.xlabel('Frame')
        plt.ylabel('Translational drift [pixels]')
    
        plt.tight_layout()
        plt.show()
        
    return janus_nodrift, tracer_nodrift