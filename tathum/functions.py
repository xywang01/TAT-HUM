"""
A series of functions directly obtained from the Trajectory class. These functions allow the user to manually specify
how they would want their data to be processed with more flexibility.

Written by X.M. Wang.

Wang, X.M., & Welsh, T.N. (2023). TAT-HUM: Trajectory Analysis Toolkit for Human Movements in Python.

"""

from typing import Union
import vg
import numpy as np
from scipy import interpolate
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation
from skspatial.objects import Plane, Points, Vector
from pytransform3d.rotations import matrix_from_axis_angle


def check_input_coord(x, y, z):
    """
    a helper function to check the dimensions of the input x, y, and z coordinates
    """
    if not x.shape == y.shape == z.shape:
        raise ValueError('input x, y, and z have to be the same shape!')

    if len(x.shape) > 1:
        x = x.squeeze()
        y = y.squeeze()
        z = z.squeeze()

    return x, y, z


def consecutive(data, step_size=1):
    """ splits the missing data indices into chunks"""
    return np.split(data, np.where(np.diff(data) != step_size)[0] + 1)


def fill_missing_data(x: np.ndarray,
                      y: np.ndarray,
                      z: np.ndarray,
                      time: np.ndarray,
                      missing_data_value=0.):
    """
    identify and fill missing data using Scipy's interp1d function

    :param x: position along the x-axis
    :param y: position along the y-axis
    :param z: position along the z-axis
    :param time: the time vector
    :param missing_data_value: the default value for missing data points
    :return: x, y, z, time, missing_info = {contain_missing, n_missing, missing_ind}
    """
    x, y, z = check_input_coord(x, y, z)

    missing_ind = np.where(x == missing_data_value)[0]
    not_missing_ind = np.where(x != missing_data_value)[0]

    if len(missing_ind) > 0:

        f_interp = interpolate.interp1d(time[not_missing_ind], x[not_missing_ind], bounds_error=False,
                                        # fill values are for missing values outside the range of x
                                        fill_value=(np.NaN, np.NaN))
        x = f_interp(time)

        f_interp = interpolate.interp1d(time[not_missing_ind], y[not_missing_ind], bounds_error=False,
                                        fill_value=(np.NaN, np.NaN))
        y = f_interp(time)

        f_interp = interpolate.interp1d(time[not_missing_ind], z[not_missing_ind], bounds_error=False,
                                        fill_value=(np.NaN, np.NaN))
        z = f_interp(time)

        ind_delete = np.where(np.isnan(x))[0]
        x = np.delete(x, ind_delete)
        y = np.delete(y, ind_delete)
        z = np.delete(z, ind_delete)
        time = np.delete(time, ind_delete)

        missing_info = {
            'contain_missing': True,
            'n_missing': len(missing_ind),
            'missing_ind': missing_ind
        }

        return x, y, z, time, missing_info
    else:
        missing_info = {
            'contain_missing': False,
            'n_missing': 0,
            'missing_ind': []
        }

        return x, y, z, time, missing_info


def low_butter(signal, fs, fc, order=2):
    """
    Direct usage of the low-pass Butterworth Filter using library from SciPy.

    :param signal: 1D data to be filtered
    :param fs: sampling frequency
    :param fc: cutoff frequency
    :param order: butterworth order
    :return: filtered signal
    """

    Wn = fc / (fs / 2)
    poly = butter(order, Wn, btype='lowpass', output='ba')  # returns numerator [0] and denominator [1] polynomials
    return filtfilt(poly[0], poly[1], signal.copy())


def compute_transformation(surface_points_x: np.ndarray,
                           surface_points_y: np.ndarray,
                           surface_points_z: np.ndarray,
                           horizontal_norm_name: str = 'y',
                           primary_ax_name: str = 'z',
                           secondary_ax_name: str = 'x',
                           full_output: bool = False) -> (Rotation, np.ndarray, dict):
    """
    :param surface_points_x: a vector specifying the x coordinates of points sampled from a surface
    :param surface_points_y: a vector specifying the y coordinates of points sampled from a surface
    :param surface_points_z: a vector specifying the z coordinates of points sampled from a surface
    :param horizontal_norm_name: the name of the axis that is perpendicular to the horizontal plane
    :param primary_ax_name: the name of the axis that corresponds to the primary movement direction
    :param secondary_ax_name: the name of the axis that corresponds to the secondary movement direction
    :param full_output: whether to return full output, which includes the objects for the plane and corners
    :return:
    """
    def find_unit_ax(ax: str):
        if ax == 'x':
            return Vector(vg.basis.x), 0
        elif ax == 'y':
            return Vector(vg.basis.y), 1
        elif ax == 'z':
            return Vector(vg.basis.z), 2
        else:
            raise ValueError('the axis has to be "x", "y", or "z"!')

    primary_norm, _ = find_unit_ax(primary_ax_name)
    secondary_norm, _ = find_unit_ax(secondary_ax_name)
    horizontal_norm, horizontal_ind = find_unit_ax(horizontal_norm_name)

    # concatenate the points
    surface_points = np.concatenate([np.expand_dims(surface_points_x, axis=1),
                                     np.expand_dims(surface_points_y, axis=1),
                                     np.expand_dims(surface_points_z, axis=1)], axis=1)

    # center the points before fitting a plane through it
    surface_center = np.mean(surface_points, axis=0)
    surface_points = surface_points - surface_center

    # find the best fitting plane and its surface normal
    surface_plane = Plane.best_fit(surface_points)
    surface_norm = Vector(surface_plane.cartesian()[:3])

    # project the current surface normal to the horizontal plane
    screen_norm_ground = surface_norm - horizontal_norm.project_vector(surface_norm)

    # find the angle between the projected norm and the primary direction - this is to align the screen's primary
    # direction with the primary direction in the Cartesian coordinate
    angle = -screen_norm_ground.angle_between(primary_norm)
    # make sure the angle is less than 90°
    angle = np.pi - angle if np.abs(angle) > np.pi / 2 else angle

    # construct the rotation matrix and rotate the surface normal
    rotmat = matrix_from_axis_angle(np.hstack((horizontal_norm, (angle,))))
    rotation_to_align = Rotation.from_matrix(rotmat)
    screen_norm_rot = Vector(rotation_to_align.apply(surface_norm))

    # after the screen is aligned with the primary axis, we can rotate it around the secondary axis to make the
    # screen surface horizontal
    angle = screen_norm_rot.angle_between(horizontal_norm)

    # axis is around the secondary direction
    rotmat = matrix_from_axis_angle(np.hstack((secondary_norm, (angle,))))
    rotation_to_ground = Rotation.from_matrix(rotmat)

    # the complete transformation - need to rotate to align first, then rotate to ground
    rotation = rotation_to_ground * rotation_to_align

    # rotate the surface points to transform them back to horizontal
    surface_points_rot = Points(rotation.apply(surface_points)) + surface_center
    screen_plane_rot = Plane.best_fit(surface_points_rot)

    if full_output:
        return rotation, surface_center, {
            'surface_plane': surface_plane,
            'surface_points_rot': surface_points_rot,
            'surface_plane_rot': screen_plane_rot,
        }
    else:
        return rotation, surface_center


def rotate_coord(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 surface_center: np.ndarray, rotation: Rotation) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param surface_center: the center of the surface, obtained through compute_transformation
    :param rotation: the Rotation object obtained through compute_transformation
    :return: rotated coordinates
    """

    coord = np.concatenate([np.expand_dims(x, axis=1),
                            np.expand_dims(y, axis=1),
                            np.expand_dims(z, axis=1)], axis=1)

    coord -= surface_center  # center the points before applying the rotation
    coord_rot = rotation.apply(coord)
    return coord_rot[:, 0], coord_rot[:, 1], coord_rot[:, 2]


def find_movement_bounds(velocity: np.ndarray,
                         velocity_threshold: Union[float, int],
                         allow_multiple_segments=False) -> (int, int):
    """
    :param velocity: a numpy ndarray (1D or 2D). If the input velocity is 2D, then the Pythagorean of the velocities
    across different dimensions will be computed and used to determine movement.
    :param velocity_threshold: the threshold based on which the movement initiation and termination will be determined.
    :param allow_multiple_segments: whether returning more than one segments or automatically identify the segment with
    the most data points
    :return: the movement initiation and termination indices, either a single index for each or a list of indices

    Find movement start and end indices.
    """
    vel_dim = velocity.shape

    if len(vel_dim) == 1:
        # only one coordinate was supplied, thus taking the absolute value of the velocity. This is to prevent
        # situations when negative velocity is not considered as exceeding the velocity threshold.
        vel_eval = np.abs(velocity)
    elif len(vel_dim) == 2:
        # when more than one coordinates were supplied, use the pythagorean of all the supplied coordinates
        vel_eval = np.linalg.norm(velocity, axis=int(np.argmin(vel_dim)))
    else:
        raise ValueError("The input velocity has to be a 1D or a 2D Numpy array!")

    n_frames = len(vel_eval)

    vel_threshold_ind = np.where(vel_eval >= velocity_threshold)[0]

    if len(vel_threshold_ind) == 0:
        # in case there's no movement detected
        return np.array([np.nan, np.nan])
    else:
        # see if more than one movement segment is identified
        vel_ind = consecutive(vel_threshold_ind)

        # directly return the indices if there is only one segments
        if len(vel_ind) == 1:
            move_start_ind = vel_threshold_ind[0] - 1 if vel_threshold_ind[0] - 1 > 0 else 0
            move_end_ind = vel_threshold_ind[-1] + 1 if vel_threshold_ind[-1] + 1 < n_frames - 1 else n_frames - 1
            return move_start_ind, move_end_ind

        if allow_multiple_segments:
            # when there are more than one segments and the user wants to save all segments
            move_start_ind = []
            move_end_ind = []

            for segment in vel_ind:
                temp_start = segment[0]
                temp_end = segment[-1]

                move_start_ind.append(temp_start - 1 if temp_start > 0 else 0)
                move_end_ind.append(temp_end + 1 if temp_end < n_frames - 1 else n_frames - 1)
        else:
            # when there are more than one segments and the user wants to keep the segments with the most data
            vel_len = [len(vel) for vel in vel_ind]
            # only use the portion of movement with the largest number of samples
            max_vel = np.where(vel_len == np.max(vel_len))[0][0]
            vel_threshold_ind = vel_ind[max_vel]

            move_start_ind = vel_threshold_ind[0] - 1 if vel_threshold_ind[0] - 1 > 0 else 0
            move_end_ind = vel_threshold_ind[-1] + 1 if vel_threshold_ind[-1] + 1 < n_frames - 1 else n_frames - 1

        return move_start_ind, move_end_ind

def find_start_end_pos(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       ind_start: int, ind_end: int, ind_buffer: int = 20) -> (np.ndarray, np.ndarray):
    """
    :param x: position along the x axis
    :param y: position along the y axis
    :param z: position along the z axis
    :param ind_start: the movement start index, obtained using find_movement_boudns function
    :param ind_end: the movement end index, obtained using find_movement_boudns function
    :param ind_buffer: number of coordinates to take before and after the start and end positions, respectively
    :return: start and end positions
    """

    x, y, z = check_input_coord(x, y, z)

    ind_start_low_bound = ind_start - ind_buffer if ind_start - ind_buffer >= 0 else 0
    ind_end_up_bound = ind_end + ind_buffer if ind_end + ind_buffer <= len(x) - 1 else len(x) - 1

    start_x = x[ind_start_low_bound:ind_start]
    start_y = y[ind_start_low_bound:ind_start]
    start_z = z[ind_start_low_bound:ind_start]
    mean_start = np.mean(np.array([start_x, start_y, start_z]), axis=1)

    end_x = x[ind_end:ind_end_up_bound]
    end_y = y[ind_end:ind_end_up_bound]
    end_z = z[ind_end:ind_end_up_bound]
    mean_end = np.mean(np.array([end_x, end_y, end_z]), axis=1)

    return mean_start, mean_end


def cent_diff(time: np.ndarray, signal: np.ndarray):
    """
    Central difference method to find derivatives.

    :param time: the time vector
    :param signal: the signal to be smoothed
    """
    n_frames = len(time)

    if len(signal) != n_frames:
        raise ValueError('The input signal has to be of the same length as its corresponding time vector!')

    der = np.zeros(n_frames, dtype=float)

    der[0] = (signal[1] - signal[0]) / (time[1] - time[0])
    der[-1] = (signal[-1] - signal[-2]) / (time[-1] - time[-2])

    for i_frame in np.arange(1, n_frames - 1):
        der[i_frame] = (signal[i_frame + 1] - signal[i_frame - 1]) / (
                time[i_frame + 1] - time[i_frame - 1])

    return der


def b_spline_fit_1d(time_vec, coord, n_fit, smooth=0., return_spline=False):
    """
    parameterize a single coordinate dimension

    :param time_vec: vector with the time stamps
    :param coord: coordinates from a single dimension
    :param n_fit: number of data points to be fitted
    :param smooth: smoothing factor
    :param return_spline: whether to also return the spline object
    :return: time_fit, trajectory_fit
    """
    tck = interpolate.splrep(time_vec, coord,
                             s=smooth,  # smoothing factor
                             k=3,  # degree of the spline fit. Scipy recommends using cubic splines.
                             )
    time_fit = np.linspace(np.min(time_vec), np.max(time_vec), n_fit)
    spline = interpolate.BSpline(tck[0], tck[1], tck[2])

    if return_spline:
        return time_fit, spline(time_fit), spline
    else:
        return time_fit, spline(time_fit)
