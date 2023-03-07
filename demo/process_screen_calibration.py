"""
Example script to process the spatial calibration data for the spatial cueing experiment in Wang & Welsh (2023). Written
by X.M. Wang.

Wang, X.M., & Welsh, T.N. (2023). TAT-HUM: Trajectory Analysis Toolkit for Human Movements in Python.

"""

import numpy as np
import pandas as pd
from skspatial.objects import Plane, Points, Vector

from tathum.trajectory import Trajectory
import matplotlib.pyplot as plt


def process_screen_calibration(root, par_id_all, debug_plot=False, return_center=False):
    """
    Originally, this function was simply a script similar to those provided in sample_data_analysis.py. Once the script
    was tested, refactoring it into a standalone function allows cleaner code in the main analysis script.
    """

    # specify the relevant information to identify the calibration files
    screen_calib_name = 'calibration'
    calib_trial_all = (1, 2, 3, 4)

    # empty lists and a numpy array to store relevant data
    end_point_all_par = []
    center_all_par = []
    transformation_lookup = np.empty((len(par_id_all), 6))
    for count, par_id in enumerate(par_id_all):

        # create a unique list and a numpy array to store participant-specific data
        trajectories_all = []
        end_points_all = np.zeros((len(calib_trial_all), 3))

        # iterate through all calibration trials
        for trial_ind, calib_trial in enumerate(calib_trial_all):
            # read the data
            calib_file_name = f'{root}/par{par_id}/par{par_id}_{screen_calib_name}_{calib_trial}.csv'
            calib_data = pd.read_csv(calib_file_name, delimiter=',')
            calib_data.columns = ('time', 'frame', 'x', 'y', 'z')

            try:
                trajectory = Trajectory(calib_data.x, calib_data.y, calib_data.z, time=calib_data.time, fs=250, fc=10)
            except:
                # because we will only need three points to specify a 3D surface, skip the trial if there are issues
                # instantiating a Trajectory object, possibly due to reasons such as empty data file (due to glitches
                # in data collection), missing movement segment (participants did not move during the data collection
                # period), etc.
                print(f'Something wrong with initializing the trajectory for the screen calibration! \n '
                      f'Participant {par_id}, trial {trial_ind}')
                continue

            # store the trajectory and the end point
            trajectories_all.append(trajectory)
            end_points_all[trial_ind, :] = trajectory.end_pos

        # remove empty end points (rows with 0 values)
        end_points_all = end_points_all[~np.all(end_points_all == 0, axis=1)]
        end_points_all = end_points_all[~np.isnan(end_points_all).any(axis=1)]

        # derive the center which will be used in the main analysis script to translate the trajectory before the
        # rotation
        end_points_center = np.mean(end_points_all, axis=0)
        center_all_par.append(end_points_center)

        # center the end points and convert them to Points, which provide accessible 3D geometrical manipulation
        end_points_all = end_points_all - end_points_center
        end_point = Points(end_points_all)
        end_point_all_par.append((par_id, end_point))

        # fit a 3D plane to the end points and identify the angle of its normal relative to the vertical direction
        screen_plane = Plane.best_fit(end_point)
        plane_coeff = Vector(screen_plane.cartesian())
        ground_coeff = Vector([0, 1, 0])
        angle = np.rad2deg(plane_coeff[:3].angle_between(ground_coeff))

        # the surface normal should always point upwards and therefore, if the angle is greater than 90°, inverting the
        # normal is needed
        if angle > 90:
            plane_coeff *= -1
            angle = np.rad2deg(plane_coeff[:3].angle_between(ground_coeff))

        # store the calibration information
        transformation_lookup[count, 0] = par_id
        transformation_lookup[count, 1] = angle
        transformation_lookup[count, 2:6] = plane_coeff

        # visualize the calibration before/after if requested
        if debug_plot:
            plt.figure()
            ax = plt.axes(projection='3d')

            for trajectory in trajectories_all:
                ax.plot(trajectory.x_movement_fit - end_points_center[0],
                        trajectory.y_movement_fit - end_points_center[1],
                        trajectory.z_movement_fit - end_points_center[2])

            end_point.plot_3d(ax, c='b', s=10, depthshade=True)
            screen_plane.plot_3d(ax, alpha=.2,
                                 lims_x=(np.min(end_points_all[:, 0]), np.max(end_points_all[:, 0])),
                                 lims_y=(np.min(end_points_all[:, 1]), np.max(end_points_all[:, 1])))

            for i_end, point in enumerate(end_point):
                ax.text(point[0], point[1], point[2], i_end)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title(f'par id = {par_id}; angle = {angle}')
            plt.show()

    if return_center:
        return end_point_all_par, center_all_par
    else:
        return end_point_all_par
