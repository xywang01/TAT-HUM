"""
A series of functions directly obtained from the Trajectory class. These functions allow the user to manually specify
how they would want their data to be processed with more flexibility.

Written by X.M. Wang.

Wang, X.M., & Welsh, T.N. (2023). TAT-HUM: Trajectory Analysis Toolkit for Human Movements in Python.
"""

from typing import Union, Optional
import vg
import numpy as np
from scipy import interpolate
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation
from skspatial.objects import Plane, Points, Vector
from enum import Enum
from functools import reduce


def check_input_coord_2d(x, y):
    """
    a helper function to check the dimensions of the input x and y coordinates
    """
    if not x.shape == y.shape:
        raise ValueError('input x and y have to be the same shape!')

    if len(x.shape) > 1:
        x = x.squeeze()
        y = y.squeeze()

    return x, y


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


# def find_missing_data(coord: np.ndarray,
#                       missing_data_value=0.):
#     """
#     identify the missing data points using any coordinate
#
#     :param x: position along the x-axis
#     :param y: position along the y-axis
#     :param z: position along the z-axis
#     :param missing_data_value: the default value for missing data points
#     :return: x, y, z, missing_info = {contain_missing, n_missing, missing_ind}
#     """
#
#     missing_ind = np.where(coord == missing_data_value)[0]
#     not_missing_ind = np.where(coord != missing_data_value)[0]
#     contain_missing = len(missing_ind) > 0
#     return contain_missing, missing_ind, not_missing_ind


def fill_missing_data(x: np.ndarray,
                      y: np.ndarray,
                      time: np.ndarray,
                      z: Optional[np.ndarray] = None,
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

    if z is None:
        x, y = check_input_coord_2d(x, y)
    else:
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

        if z is not None:
            f_interp = interpolate.interp1d(time[not_missing_ind], z[not_missing_ind], bounds_error=False,
                                            fill_value=(np.NaN, np.NaN))
            z = f_interp(time)

        ind_delete = np.where(np.isnan(x))[0]
        x = np.delete(x, ind_delete)
        y = np.delete(y, ind_delete)
        if z is not None:
            z = np.delete(z, ind_delete)
        time = np.delete(time, ind_delete)

        # identify the consecutive missing data points
        missing_ind_segments = consecutive(missing_ind)
        n_missing_segments = [len(seg) for seg in missing_ind_segments]

        missing_info = {
            'contain_missing': True,
            'n_missing': len(missing_ind),
            'missing_ind': missing_ind,
            'missing_ind_segments': missing_ind_segments,
            'n_missing_segments': n_missing_segments
        }

        if z is None:
            return x, y, time, missing_info
        else:
            return x, y, z, time, missing_info
    else:
        missing_info = {
            'contain_missing': False,
            'n_missing': 0,
            'missing_ind': [],
            'missing_ind_segments': [],
            'n_missing_segments': []
        }

        if z is None:
            return x, y, time, missing_info
        else:
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


def find_optimal_cutoff_frequency(signal: np.ndarray,
                                  fs: int,
                                  fc_test: np.ndarray = np.arange(2, 14),
                                  ):
    """
    Identify the optimal cutoff frequency based on the normalized residual autocorrelation (see Cappello, La
    Palombara, & Leardini, 1996)

    :param signal: the signal to be filtered
    :param fs: sampling frequency
    :param fc_test: the range of cutoff frequencies to be tested
    Default range is obtained from Schreven, Beek, & Smeets (2015). However, according to Barlett (2007), cutoff
    frequencies between 4 and 8 Hz are often used in filtering movement data. Decided to use a wider range.
    :return: the optimal cutoff frequency
    """

    def autocorr(_signal):
        _signal_mean = np.mean(_signal)

        # Compute the autocorrelation using NumPy's correlate function
        _corr = np.correlate(_signal - _signal_mean, _signal - _signal_mean, mode='full')

        # Normalize the autocorrelation
        _corr /= (np.std(_signal) ** 2) * len(_signal)

        # based on Cappello, La Palombara, & Leardini (1996), only use up to 10 lags
        _corr = _corr[:10]

        # get the summed squares of the autocorrelation
        _corr = np.sum(_corr ** 2)

        return _corr

    norm_resid = []

    # compute the normalized residual autocorrelation for each cutoff frequency
    for fc in fc_test:
        # compute the residual
        resid = signal - low_butter(signal, fs, fc)
        # compute the normalized residual autocorrelation
        resid_autocorr = autocorr(resid)

        norm_resid.append(resid_autocorr)
    norm_resid = np.array(norm_resid)

    # return the cutoff frequency with the smallest normalized residual autocorrelation
    return fc_test[np.argmin(norm_resid)]


def rotation_mat_2d(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]])

def compute_transformation_2d(start_pos: np.ndarray,
                              end_pos: np.ndarray,
                              to_dir: np.ndarray = np.array([0, 1]),):
    # normalized vector between the start and end positions
    movement_ax = end_pos - start_pos
    movement_ax_norm = movement_ax / np.linalg.norm(movement_ax)

    # get the angle between the desired direction and the movement axis
    angle = np.arccos(np.matmul(movement_ax_norm, to_dir))

    # get the rotational matrix based on the angle
    return rotation_mat_2d(angle)

def compute_transformation_3d(surface_points_x: np.ndarray,
                              surface_points_y: np.ndarray,
                              surface_points_z: np.ndarray,
                              horizontal_norm: Union[str, int] = 'y',
                              primary_ax: Union[str, int] = 'z',
                              secondary_ax: Union[str, int] = 'x',
                              full_output: bool = False) -> (Rotation, np.ndarray, dict):
    """
    :param surface_points_x: a vector specifying the x coordinates of points sampled from a surface
    :param surface_points_y: a vector specifying the y coordinates of points sampled from a surface
    :param surface_points_z: a vector specifying the z coordinates of points sampled from a surface
    :param horizontal_norm: the name of the axis that is perpendicular to the horizontal plane
    :param primary_ax: the name of the axis that corresponds to the primary movement direction
    :param secondary_ax: the name of the axis that corresponds to the secondary movement direction
    :param full_output: whether to return full output, which includes the objects for the plane and corners
    :return:
    """
    def find_unit_ax(ax: Union[str, int]):
        if (ax == 'x') | (ax == 0):
            return Vector(vg.basis.x), 0
        elif (ax == 'y') | (ax == 1):
            return Vector(vg.basis.y), 1
        elif (ax == 'z') | (ax == 2):
            return Vector(vg.basis.z), 2
        else:
            raise ValueError('the axis has to be "x", "y", or "z"!')

    def rotate_vec_to_target(_vec, _target):
        # Normalize the input normal vector
        _vec = _vec / np.linalg.norm(_vec)

        # Calculate the axis of rotation (cross product between normal and target)
        axis = np.cross(_vec, _target)

        # Calculate the cosine of the angle between the two vectors
        cos_angle = np.dot(_vec, _target)

        # Calculate the skew-symmetric cross-product matrix
        cross_product_matrix = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        # Create the rotation matrix using the Rodrigues' formula
        rotation_matrix = np.identity(3) + cross_product_matrix + np.dot(cross_product_matrix, cross_product_matrix) * (
                    1 - cos_angle) / (np.linalg.norm(axis) ** 2)

        return rotation_matrix

    def project_vector_onto_plane(_vec, _plane_normal):
        # Normalize the plane normal vector
        _plane_normal = _plane_normal / np.linalg.norm(_plane_normal)

        # Calculate the projection of the vector onto the plane
        projection = _vec - np.dot(_vec, _plane_normal) * _plane_normal

        return projection

    primary_norm, _ = find_unit_ax(primary_ax)
    secondary_norm, _ = find_unit_ax(secondary_ax)
    horizontal_norm, horizontal_ind = find_unit_ax(horizontal_norm)

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
    surface_norm = surface_norm / np.linalg.norm(surface_norm)

    # project the current surface normal to the horizontal plane
    screen_norm_ground = project_vector_onto_plane(surface_norm, horizontal_norm)

    # rotation to transform the surface normal to the horizontal plane
    rotation_horizontal = Rotation.from_matrix(rotate_vec_to_target(surface_norm, horizontal_norm))

    # rotation to align the surface orientation to the primary axis
    rotation_primary = Rotation.from_matrix(rotate_vec_to_target(screen_norm_ground, primary_norm))

    # combine the two rotations
    rotation = rotation_primary * rotation_horizontal

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


def find_movement_bounds(movement_feature: np.ndarray,
                         feature_threshold: Union[float, int],
                         allow_multiple_segments=False) -> (int, int):
    """
    :param movement_feature: a numpy ndarray (1D or 2D) that will be used to evaluate against a threshold. This could
    be velocity or acceleration. If the input velocity is 2D, then the Pythagorean of the velocities across different
    dimensions will be computed and used to determine movement.
    :param feature_threshold: the threshold based on which the movement initiation and termination will be determined.
    :param allow_multiple_segments: whether returning more than one segments or automatically identify the segment with
    the most data points
    :return: the movement initiation and termination indices, either a single index for each or a list of indices

    Find movement start and end indices.
    """
    dim = movement_feature.shape

    if len(dim) == 1:
        # only one coordinate was supplied, thus taking the absolute value of the velocity. This is to prevent
        # situations when negative velocity is not considered as exceeding the velocity threshold.
        feature_eval = np.abs(movement_feature)
    elif len(dim) == 2:
        # when more than one coordinates were supplied, use the pythagorean of all the supplied coordinates
        feature_eval = np.linalg.norm(movement_feature, axis=int(np.argmin(dim)))
    else:
        raise ValueError("The input velocity has to be a 1D or a 2D Numpy array!")

    n_frames = len(feature_eval)

    feature_threshold_ind = np.where(feature_eval >= feature_threshold)[0]

    if len(feature_threshold_ind) == 0:
        # in case there's no movement detected
        return np.array([np.nan, np.nan])
    else:
        # see if more than one movement segment is identified
        feature_ind = consecutive(feature_threshold_ind)

        # directly return the indices if there is only one segments
        if len(feature_ind) == 1:
            move_start_ind = feature_threshold_ind[0] - 1 if feature_threshold_ind[0] - 1 > 0 else 0
            move_end_ind = feature_threshold_ind[-1] + 1 if feature_threshold_ind[-1] + 1 < n_frames - 1 else n_frames - 1
            return move_start_ind, move_end_ind

        if allow_multiple_segments:
            # when there are more than one segments and the user wants to save all segments
            move_start_ind = []
            move_end_ind = []

            for segment in feature_ind:
                temp_start = segment[0]
                temp_end = segment[-1]

                move_start_ind.append(temp_start - 1 if temp_start > 0 else 0)
                move_end_ind.append(temp_end + 1 if temp_end < n_frames - 1 else n_frames - 1)
        else:
            # when there are more than one segments and the user wants to keep the segments with the most data
            feature_len = [len(feature) for feature in feature_ind]
            # only use the portion of movement with the largest number of samples
            max_feature = np.where(feature_len == np.max(feature_len))[0][0]
            feature_threshold_ind = feature_ind[max_feature]

            move_start_ind = feature_threshold_ind[0] - 1 if feature_threshold_ind[0] - 1 > 0 else 0
            move_end_ind = feature_threshold_ind[-1] + 1 if feature_threshold_ind[-1] + 1 < n_frames - 1 else n_frames - 1

        return move_start_ind, move_end_ind


def find_movement_bounds_percent_threshold(feature, percent_feature=0.05, allow_multiple_segments=False):
    """ Percentage of peak velocity as the threshold to find movement bounds. """
    peak_vel = np.max(np.abs(feature))
    vel_threshold = percent_feature * peak_vel
    return find_movement_bounds(feature, vel_threshold, allow_multiple_segments)


def find_movement_bounds_displacement(displacement: np.ndarray,
                                      pos_start: np.ndarray,
                                      pos_end: np.ndarray,
                                      threshold: float = 0.1):
    # get the Euclidean distance between the displacement trajectory and start position
    dist_start = np.linalg.norm(displacement - pos_start, axis=1)
    # get the Euclidean distance between the displacement trajectory and end position
    dist_end = np.linalg.norm(displacement - pos_end, axis=1)

    # find the indices where the distance is smaller than the threshold
    start_ind = np.where(dist_start < threshold)[0]
    end_ind = np.where(dist_end < threshold)[0]

    # return the start and end indices
    return start_ind[0], end_ind[-1]

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


class Preprocesses(Enum):
    FILL_MISSING = 1
    SPATIAL_TRANSFORM = 2
    LOW_BUTTER = 3
    CENT_DIFF = 4


def composite_function(*functions):
    def compose(f, g):
        return lambda x: f(g(x))

    return reduce(compose, functions, lambda x: x)
