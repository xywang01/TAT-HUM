"""
The Trajectory class that takes the raw trajectory data and automatically processes the data to derive relevant action
measures.

Written by X.M. Wang.

Wang, X.M., & Welsh, T.N. (2023). TAT-HUM: Trajectory Analysis Toolkit for Human Movements in Python.
"""

import typing
import pandas as pd
from .coord import Coord
import numpy as np
from skspatial.objects import Plane
import matplotlib.pyplot as plt
from .trajectory_base import TrajectoryBase
from .functions import Preprocesses, cent_diff, low_butter, fill_missing_data, find_optimal_cutoff_frequency, \
    compute_transformation_3d


class Trajectory(TrajectoryBase):
    """
    Automatically processes trajectory data.
    """
    x = Coord()
    y = Coord()
    z = Coord()
    time = Coord()

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray,

                 displacement_preprocess: typing.Tuple[Preprocesses, ...] = (Preprocesses.LOW_BUTTER,),
                 velocity_preprocess: typing.Tuple[Preprocesses, ...] = (Preprocesses.CENT_DIFF,),
                 acceleration_preprocess: typing.Tuple[Preprocesses, ...] = (Preprocesses.CENT_DIFF,),

                 transform_end_points=None,  # end points used for spatial transformation

                 time: typing.Optional[np.ndarray] = None,

                 movement_plane_ax='xz',  # 2D plane that specifies the principal direction of the reach
                 primary_dir='z',  # the primary movement direction
                 ground_dir='y',  # the ground's normal direction

                 movement_selection_ax='z',
                 movement_selection_method='length',
                 movement_selection_sign=None,
                 custom_compute_movement_boundary: typing.Optional[callable] = None,

                 center_movement=True,  # whether to center the movement at the origin

                 unit: str = 'mm',
                 missing_data_value: float = 0.,
                 fs: typing.Optional[int] = None,
                 fc: typing.Optional[int] = None,
                 vel_threshold: float = 50.,

                 movement_pos_time_cutoff=0.2,  # used for finding the start and end positions

                 spline_order=3,
                 n_spline_fit=100,
                 ):
        """
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate

        :param displacement_preprocess: the preprocesses to apply to the displacement data. See functions.Preprocesses
        for the available preprocesses.
        :param velocity_preprocess: the preprocesses to apply to the velocity data. See functions.Preprocesses for the
        available preprocesses.
        :param acceleration_preprocess: the preprocesses to apply to the acceleration data. See functions.Preprocesses
        for the available preprocesses.

        :param transform_end_points: A minimum of three 3D points that specifies the plane on which the movement
        occurred. If provided, the best-fitting plane for these points will be computed and used to transform the
        trajectory so that the transformed movement would take place on a horizontal surface. This could be derived
        using the same Trajectory class.

        :param time: the corresponding time stamps

        :param movement_plane_ax: The 2D plane (xy, xz, or yz) of interest.
        :param primary_dir: The primary direction of the movement. The primary direction is the direction of the
        movement that is of most theoretical relevance to the experiment. For example, if the experiment is about
        reaching, then the primary direction is the direction of the reach. The secondary direction would be deduced
        based on the primary direction and the movement plane.
        :param ground_dir: The ground's normal direction (upright). This is used to determine the transformation matrix.

        :param movement_selection_ax: The axis ('x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz') along which the movement
        selection should be performed.
        :param movement_selection_method: The method to select the movement segment in case extraneous movements are
        detected. Can be chosen from ('length', 'sign', ). 'length' simply select the movement segment with the most
        number of points, whereas 'sign' selects the movement segment based on movement direction. 'length' is more
        effective for small extraneous movements whereas 'sign' is more effective for movement reversals during the data
        collection period. See movement_selection_sign for more.
        :param movement_selection_sign: Can be chosen from ('positive', 'negative'), only applies to directional
        movement. The sign is determined by the difference between the end and start position.
        :param custom_compute_movement_boundary: A custom function to compute the movement initiation and termination
        boundaries. The function should take in the trajectory object and return a tuple of size 3 for
        1) time_start (float, np.float64), 2) time_end (float, np.float64),
        3) movement indices (list[(int, np.ind64)], np.ndarray[(int, np.ind64)])

        :param center_movement: whether to center the movement at the origin

        :param unit: position measure's unit
        :param missing_data_value: the filler value for missing data, default to be 0. This would be used for missing
        data interpolation.
        :param fs: sampling frequency (Hz). This can be left as None if the time stamps are provided.
        :param fc: cutoff frequency (for the Butterworth filter). This can be left as None if the cutoff frequency is
        to be computed automatically.
        :param vel_threshold: The velocity threshold to determine movement initiation and termination, in the same unit
        as the displacement data.

        :param movement_pos_time_cutoff: The amount of time before/after the movement initiation/termination to
        consider when computing the trajectory's end positions. Instead of using the position at a particular time,
        the end position is the average of all positions within this time cutoff range.

        :param spline_order: the order of the b-spline fit, default to 3
        :param n_spline_fit: number of fitted trajectory points when using bspline
        """
        self.x_original, self.y_original, self.z_original = x.copy(), y.copy(), z.copy()  # keep a separate copy of the original data
        self.x, self.y, self.z = x, y, z
        self.x_vel, self.y_vel, self.z_vel = np.array([]), np.array([]), np.array([])
        self.x_acc, self.y_acc, self.z_acc = np.array([]), np.array([]), np.array([])
        self.n_frames = self.validate_size()

        if (time is None) & (fs is None):
            raise ValueError('You have to either specify the time stamps or the sampling frequency!')
        if time is None:
            self.time = np.linspace(0, self.n_frames * 1 / fs, num=self.n_frames, endpoint=False)
            self.time_original = self.time.copy()
        else:
            self.time_original = time.copy()
            self.time = time
            fs = 1 / np.mean(np.diff(self.time))
            if len(self.time) != self.n_frames:
                raise ValueError('The size of the input time stamps is not the same as the size of the coordinates!')

        super().__init__(
            unit=unit,
            missing_data_value=missing_data_value,
            fs=fs, fc=fc,
            vel_threshold=vel_threshold,
            movement_selection_method=movement_selection_method, movement_selection_sign=movement_selection_sign,
            custom_compute_movement_boundary=custom_compute_movement_boundary,
            spline_order=spline_order, n_spline_fit=n_spline_fit, )

        self.center_movement = center_movement

        self.movement_plane_ax = movement_plane_ax
        self.secondary_dir = None  # placeholder - will be set automatically after setting primary dir
        self.primary_dir = primary_dir
        self.ground_plane = None  # placeholder - will be set automatically after setting ground_dir
        self.ground_dir = ground_dir

        self.movement_selection_method = movement_selection_method
        self.movement_selection_sign = movement_selection_sign
        if self.movement_selection_method == 'sign':
            if self.movement_selection_sign is None:
                raise ValueError('movement_selection_method is set to "sign" but movement_selection_sign was not specified!')
            if movement_selection_ax is None:
                self.movement_selection_dir = self.primary_dir
            else:
                if movement_selection_ax == 'x':
                    self.movement_selection_dir = 0
                elif movement_selection_ax == 'y':
                    self._primary_dir = 1
                elif movement_selection_ax == 'z':
                    self._primary_dir = 2
                else:
                    raise ValueError('Invalid movement_selection_dir! Please use the following: x, y, or z')

        self.contain_movement = True  # whether there was actual movement

        # eliminate missing data; need to do it before the transformation
        self.contain_missing, self.n_missing, self.ind_missing, self.missing_segments, self.n_missing_segments = self.missing_data()

        # if the cutoff frequency is not specified, then it will be computed automatically
        if self.fc is None:
            self.fc = find_optimal_cutoff_frequency(self.x, self.fs)

        # transform the data if necessary
        self.transform_end_points = transform_end_points
        if self.transform_end_points is not None:
            self.transform_mat, self.transform_origin, transform_info = self.compute_transformation(
                self.transform_end_points, full_output=True)
            self.screen_plane = transform_info['screen_plane']
            self.screen_corners_rot = transform_info['screen_corners_rot']
            self.screen_plane_rot = transform_info['screen_plane_rot']
            self.x, self.y, self.z = self.transform_data(self.x, self.y, self.z)
            self.x_original, self.y_original, self.z_original = self.transform_data(
                self.x_original, self.y_original, self.z_original)

        # preprocess the data and obtain temporal derivatives
        self.preprocess('displacement', displacement_preprocess)
        self.preprocess('velocity', velocity_preprocess)
        self.preprocess('acceleration', acceleration_preprocess)

        # identify movement initiation and termination
        self.movement_displacement = self.find_movement_displacement(movement_selection_ax=movement_selection_ax)
        self.movement_velocity = self.find_movement_velocity(movement_selection_ax=movement_selection_ax)
        self.movement_acceleration = self.find_movement_acceleration(movement_selection_ax=movement_selection_ax)
        self.start_time, self.end_time, self.movement_ind = self.compute_movement_boundaries()
        self.contain_movement = self.validate_movement()

        if self.contain_movement:
            # check if the missing data are in the movement segment
            self.missing_segments_movement = []
            if self.contain_missing:
                # check if any of the missing indices are in the movement segment
                self.ind_missing_movement = [i for i, value in enumerate(self.ind_missing)
                                             if self.movement_ind[0] <= value <= self.movement_ind[-1]]
                self.n_missing_movement = len(self.ind_missing_movement)

                # do the same for respective missing segments
                for seg in self.missing_segments:
                    if (self.movement_ind[0] <= seg[0] <= self.movement_ind[-1]) & \
                            (self.movement_ind[0] <= seg[-1] <= self.movement_ind[-1]):
                        self.missing_segments_movement.append(seg)
            else:
                self.ind_missing_movement = []
                self.n_missing_movement = 0

            self.n_missing_segments_movement = np.array([len(seg) for seg in self.missing_segments_movement])

            self.rt = self.start_time
            self.mt = self.end_time - self.start_time
            self.start_pos, self.end_pos = self.find_start_and_end_pos(time_cutoff=movement_pos_time_cutoff)

            self.time_movement = self.time[self.movement_ind]
            self.x_movement = self.x[self.movement_ind]
            self.y_movement = self.y[self.movement_ind]
            self.z_movement = self.z[self.movement_ind]
            self.x_vel_movement = self.x_vel[self.movement_ind]
            self.y_vel_movement = self.y_vel[self.movement_ind]
            self.z_vel_movement = self.z_vel[self.movement_ind]
            self.x_acc_movement = self.x_vel[self.movement_ind]
            self.y_acc_movement = self.y_vel[self.movement_ind]
            self.z_acc_movement = self.z_vel[self.movement_ind]

            self.movement_velocity = self.movement_velocity[self.movement_ind]
            peak_vel_idx = np.argmax(self.movement_velocity)
            self.peak_vel = self.movement_velocity[peak_vel_idx]
            self.time_to_peak_vel = self.time_movement[peak_vel_idx] - self.time_movement[0]
            self.time_after_peak_vel = self.time_movement[-1] - self.time_movement[peak_vel_idx]

            self.movement_acceleration = self.movement_acceleration[self.movement_ind]
            peak_acc_idx = np.argmax(self.movement_acceleration)
            self.peak_acc = self.movement_acceleration[peak_acc_idx]
            self.time_to_peak_acc = self.time_movement[peak_acc_idx] - self.time_movement[0]
            self.time_after_peak_acc = self.time_movement[-1] - self.time_movement[peak_acc_idx]

            if self.center_movement:
                self.start_pos -= self.start_pos
                self.end_pos -= self.start_pos
                self.x_movement -= self.start_pos[0]
                self.y_movement -= self.start_pos[1]
                self.z_movement -= self.start_pos[2]

            self.time_fit, self.x_fit, self.x_spline = self.b_spline_fit_1d(self.time_movement, self.x_movement, self.n_spline_fit)
            _, self.y_fit, self.y_spline = self.b_spline_fit_1d(self.time_movement, self.y_movement, self.n_spline_fit)
            _, self.z_fit, self.z_spline = self.b_spline_fit_1d(self.time_movement, self.z_movement, self.n_spline_fit)

            # instead of normalizing the velocity and accelerations, simply use cent_diff to directly compute them
            self.x_vel_fit = cent_diff(self.time_fit, self.x_fit)
            self.y_vel_fit = cent_diff(self.time_fit, self.y_fit)
            self.z_vel_fit = cent_diff(self.time_fit, self.z_fit)
            self.x_acc_fit = cent_diff(self.time_fit, self.x_vel_fit)
            self.y_acc_fit = cent_diff(self.time_fit, self.y_vel_fit)
            self.z_acc_fit = cent_diff(self.time_fit, self.z_vel_fit)
        else:
            # in case the trajectory does not satisfy the movement initiation/termination criteria, in cases such as
            # when the participant number moved during the data collection period
            self.contain_missing = None
            self.n_missing = None
            self.rt = None
            self.mt = None
            self.start_pos, self.end_pos = None, None

            self.x_movement = None
            self.y_movement = None
            self.z_movement = None
            self.x_vel_movement = None
            self.y_vel_movement = None
            self.z_vel_movement = None
            self.x_acc_movement = None
            self.y_acc_movement = None
            self.z_acc_movement = None

            self.peak_vel = None
            self.time_to_peak_vel = None
            self.time_after_peak_vel = None

            self.peak_acc = None
            self.time_to_peak_acc = None
            self.time_after_peak_acc = None

    def find_movement_angle(self, perc_of_movement=0.2):
        """
        Find the angle of the movement based on the first x% of the movement on movement_plane_ax and primary_dir.
        :param perc_of_movement: the percentage of the movement to consider when computing the movement angle
        :return: the movement angle
        """
        if self.contain_movement:
            end_idx = int(np.round(perc_of_movement * len(self.x_movement)))

            displacement_vector = self.get_displacement_vector(self.movement_plane_ax)

            start_pos = displacement_vector[0]
            end_pos = displacement_vector[end_idx]
            movement_vector = end_pos - start_pos

            primary_dir_vector = np.zeros((3,))
            primary_dir_vector[self.primary_dir] = 1

            movement_angle = np.arccos(np.dot(movement_vector, primary_dir_vector) / np.linalg.norm(movement_vector))
            return movement_angle
        else:
            return None

    def assign_preprocess_function(self,
                                   preprocess_var: str,
                                   preprocess: Preprocesses) -> (callable, tuple):
        preprocess_order = self.find_preprocess_order(preprocess_var)
        if preprocess == Preprocesses.LOW_BUTTER:
            return self.low_butter, (preprocess_order, )
        elif preprocess == Preprocesses.CENT_DIFF:
            return self.cent_diff, (preprocess_order, )
        else:
            raise ValueError('The preprocess is not recognized!')

    def low_butter(self, low_butter_order: int = 1):
        """
        A wrapper function for the low_butter() from tathum.functions specific to the current Trajectory2D class.
        """
        if low_butter_order == 1:
            self.x = low_butter(self.x, self.fs, self.fc)
            self.y = low_butter(self.y, self.fs, self.fc)
            self.z = low_butter(self.z, self.fs, self.fc)
        elif low_butter_order == 2:
            self.x_vel = low_butter(self.x_vel, self.fs, self.fc)
            self.y_vel = low_butter(self.y_vel, self.fs, self.fc)
            self.z_vel = low_butter(self.z_vel, self.fs, self.fc)
        elif low_butter_order == 3:
            self.x_acc = low_butter(self.x_acc, self.fs, self.fc)
            self.y_acc = low_butter(self.y_acc, self.fs, self.fc)
            self.z_acc = low_butter(self.z_acc, self.fs, self.fc)
        else:
            raise ValueError('The order of the low-pass Butterworth filter has to be either 1 (for displacement), 2 '
                             '(for velocity), or 3 (for acceleration)!')

    def cent_diff(self, cent_diff_order: int = 2):
        """
        A wrapper function for the cent_diff() from tathum.functions specific to the current Trajectory2D class.
        """
        if cent_diff_order == 2:
            self.x_vel = cent_diff(self.time, self.x)
            self.y_vel = cent_diff(self.time, self.y)
            self.z_vel = cent_diff(self.time, self.z)
        elif cent_diff_order == 3:
            self.x_acc = cent_diff(self.time, self.x_vel)
            self.y_acc = cent_diff(self.time, self.y_vel)
            self.z_acc = cent_diff(self.time, self.z_vel)
        else:
            raise ValueError('The order of the central difference has to be either 1 (for velocity ) or 2 '
                             '(for acceleration!')

    def get_displacement_vector(self, ax):
        """
        Get the displacement vector along the specified axis.
        :param ax: the axis ('x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz') along which the displacement vector should be
        :return: the displacement vector
        """
        axis_mapping = {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'xy': np.concatenate([np.expand_dims(self.x, axis=1), np.expand_dims(self.y, axis=1)], axis=1),
            'xz': np.concatenate([np.expand_dims(self.x, axis=1), np.expand_dims(self.z, axis=1)], axis=1),
            'yz': np.concatenate([np.expand_dims(self.y, axis=1), np.expand_dims(self.z, axis=1)], axis=1),
            'xyz': np.concatenate([
                np.expand_dims(self.x, axis=1),
                np.expand_dims(self.y, axis=1),
                np.expand_dims(self.z, axis=1)], axis=1),
        }
        return axis_mapping[ax]

    def find_movement_displacement(self, movement_selection_ax: str = 'z'):
        """
        Find the displacement vectors based on which the movement selection will be performed.
        :param movement_selection_ax: the axis ('x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz') along which the movement will
        be selected.
        :return: the displacement vectors
        """
        if movement_selection_ax in ['x', 'y', 'z']:
            return np.abs(self.get_displacement_vector(movement_selection_ax))
        elif movement_selection_ax in ['xy', 'xz', 'yz', 'xyz']:
            return np.linalg.norm(self.get_displacement_vector(movement_selection_ax), axis=1)
        else:
            raise ValueError('Invalid movement_selection_ax! Please use the following: x, y, z, xy, xz, yz, or xyz')

    def get_velocity_vector(self, ax):
        """
        Get the velocity vector along the specified axis.
        :param ax: the axis ('x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz') along which the velocity vector should be
        :return: the velocity vector
        """
        axis_mapping = {
            'x': self.x_vel,
            'y': self.y_vel,
            'z': self.z_vel,
            'xy': np.concatenate([np.expand_dims(self.x_vel, axis=1), np.expand_dims(self.y_vel, axis=1)], axis=1),
            'xz': np.concatenate([np.expand_dims(self.x_vel, axis=1), np.expand_dims(self.z_vel, axis=1)], axis=1),
            'yz': np.concatenate([np.expand_dims(self.y_vel, axis=1), np.expand_dims(self.z_vel, axis=1)], axis=1),
            'xyz': np.concatenate([
                np.expand_dims(self.x_vel, axis=1),
                np.expand_dims(self.y_vel, axis=1),
                np.expand_dims(self.z_vel, axis=1)], axis=1),
        }
        return axis_mapping[ax]

    def find_movement_velocity(self, movement_selection_ax: str = 'z'):
        """
        Find the velocity vectors based on which the movement selection will be performed.
        :param movement_selection_ax: the axis ('x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz') along which the movement will
        be selected.
        :return: the velocity vectors
        """
        if movement_selection_ax in ['x', 'y', 'z']:
            return np.abs(self.get_velocity_vector(movement_selection_ax))
        elif movement_selection_ax in ['xy', 'xz', 'yz', 'xyz']:
            return np.linalg.norm(self.get_velocity_vector(movement_selection_ax), axis=1)
        else:
            raise ValueError('Invalid movement_selection_ax! Please use the following: x, y, z, xy, xz, yz, or xyz')


    def get_acceleration_vector(self, ax):
        """
        Get the acceleration vector along the specified axis.
        :param ax: the axis ('x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz') along which the acceleration vector should be
        :return: the acceleration vector
        """
        axis_mapping = {
            'x': self.x_acc,
            'y': self.y_acc,
            'z': self.z_acc,
            'xy': np.concatenate([np.expand_dims(self.x_acc, axis=1), np.expand_dims(self.y_acc, axis=1)], axis=1),
            'xz': np.concatenate([np.expand_dims(self.x_acc, axis=1), np.expand_dims(self.z_acc, axis=1)], axis=1),
            'yz': np.concatenate([np.expand_dims(self.y_acc, axis=1), np.expand_dims(self.z_acc, axis=1)], axis=1),
            'xyz': np.concatenate([
                np.expand_dims(self.x_acc, axis=1),
                np.expand_dims(self.y_acc, axis=1),
                np.expand_dims(self.z_acc, axis=1)], axis=1),
        }
        return axis_mapping[ax]

    def find_movement_acceleration(self, movement_selection_ax: str = 'z'):
        """
        Find the acceleration vectors based on which the movement selection will be performed.
        :param movement_selection_ax: the axis ('x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz') along which the movement will
        be selected.
        :return: the acceleration vectors
        """
        if movement_selection_ax in ['x', 'y', 'z']:
            return np.abs(self.get_acceleration_vector(movement_selection_ax))
        elif movement_selection_ax in ['xy', 'xz', 'yz', 'xyz']:
            return np.linalg.norm(self.get_acceleration_vector(movement_selection_ax), axis=1)
        else:
            raise ValueError('Invalid movement_selection_ax! Please use the following: x, y, z, xy, xz, yz, or xyz')

    def compute_transformation(self, screen_corners, full_output=False):
        """
        :param screen_corners: the corners of the surface on which the movement was performed
        :param full_output: whether to return full output, which includes the objects for the plane and corners
        :return: the transformation matrix, the origin of the transformation, and the transformed screen corners
        """
        screen_center = np.mean(screen_corners, axis=0)
        rotation, surface_center, transform_info = compute_transformation_3d(
            screen_corners[:, 0], screen_corners[:, 1], screen_corners[:, 2],
            horizontal_norm=self.ground_dir,
            primary_ax=self.primary_dir,
            secondary_ax=self.secondary_dir,
            full_output=True,
        )

        if full_output:
            return rotation, screen_center, {
                'screen_plane': transform_info['surface_plane'],
                'screen_corners_rot': transform_info['surface_points_rot'],
                'screen_plane_rot': transform_info['surface_plane_rot'],
            }
        else:
            return rotation, screen_center

    def transform_data(self, x, y, z):
        """
        Spatially transform the trajectory so that the trajectory is on a flat, horizontal plane.
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        :return: transformed coordinates
        """
        coord = np.concatenate([np.expand_dims(x, axis=1),
                                np.expand_dims(y, axis=1),
                                np.expand_dims(z, axis=1)], axis=1)
        coord -= self.transform_origin
        coord_rot = self.transform_mat.apply(coord) + self.transform_origin
        return coord_rot[:, 0], coord_rot[:, 1], coord_rot[:, 2]

    def find_start_and_end_pos(self, time_cutoff):
        """
        Find the start and end positions of the movement.
        :param time_cutoff: The amount of time before/after the movement initiation/termination to consider when
        computing the trajectory's end positions. Instead of using the position at a particular time, the end position
        is the average of all positions within this time cutoff range.
        :return: mean_start, mean_end: the start and end positions
        """

        ind_start = (self.time < self.start_time + time_cutoff) & (self.time > self.start_time - time_cutoff)
        if np.any(ind_start):
            start_x = self.x[ind_start]
            start_y = self.y[ind_start]
            start_z = self.z[ind_start]
            mean_start = np.mean(np.array([start_x, start_y, start_z]), axis=1)
        else:
            mean_start = np.empty((3,)) * np.nan

        ind_end = (self.time < self.end_time + time_cutoff) & (self.time > self.end_time - time_cutoff)
        if np.any(ind_end) > 0:
            end_x = self.x[ind_end]
            end_y = self.y[ind_end]
            end_z = self.z[ind_end]
            mean_end = np.mean(np.array([end_x, end_y, end_z]), axis=1)
        else:
            mean_end = np.empty((3,)) * np.nan

        return mean_start, mean_end

    @staticmethod
    def consecutive(data, step_size=1):
        """ splits the missing data indices into chunks"""
        return np.split(data, np.where(np.diff(data) != step_size)[0] + 1)

    # def find_missing_data(self):
    #     return find_missing_data(self.x, self.missing_data_value)

    def missing_data(self):
        """
        Find if there is any outliers in the original movement trajectory. The outliers are detected by comparing each
        consecutive points' difference with a certain proportion of the overall range of motion.

        threshold: the threshold for the difference of x, y, and z, before and after the missing data block. If the
        difference is small, then we will use linear interpolation to fill in the gap. By default, the threshold for
        each axis is 1 mm/s.
        """
        self.x, self.y, self.z, self.time, missing_info = fill_missing_data(
            x=self.x, y=self.y, z=self.z, time=self.time, missing_data_value=self.missing_data_value)
        self.n_frames = self.validate_size()  # remember to update n_frames
        return missing_info['contain_missing'], missing_info['n_missing'], missing_info['missing_ind'], missing_info['missing_ind_segments'], missing_info['n_missing_segments']

    def debug_plots(self, fig=None, axs=None):
        """ Create a debug plot that shows displacement, velocity, acceleration, and XY trajectory"""
        if axs is None:
            fig, axs = plt.subplots(2, 1)
            # plt.tight_layout()

        axs[0].plot(self.time_original, self.x_original, label='x', linestyle=':')
        axs[0].plot(self.time_original, self.y_original, label='y', linestyle=':')
        axs[0].plot(self.time_original, self.z_original, label='z', linestyle=':')
        axs[0].scatter(self.time_original[self.ind_missing], self.x_original[self.ind_missing], color='k')

        axs[0].plot(self.time, self.x, label='x', c='r')
        axs[0].plot(self.time, self.y, label='y', c='g')
        axs[0].plot(self.time, self.z, label='z', c='b')

        if self.contain_missing:
            for i_seg, seg in enumerate(self.missing_segments_movement):
                n_missing = self.n_missing_segments_movement[i_seg]
                seg_mid = int(np.median(seg) - 1)

                axs[0].text(self.time[seg_mid], np.min([self.x_original, self.y_original, self.z_original]),
                            f'{n_missing}', fontsize=12,)

        for principal_dir in self.movement_plane_ax:
            axs[0].plot(self.time, np.ones((len(self.time), 1)) * self.start_pos[principal_dir],
                        c='c', linestyle=':', linewidth=3)
            axs[0].plot(self.time, np.ones((len(self.time), 1)) * self.end_pos[principal_dir],
                        c='m', linestyle=':', linewidth=3)

        axs[0].plot([self.start_time, self.start_time],
                    [np.min([self.x, self.y, self.z]),
                     np.max([self.x, self.y, self.z])], label='start', c='c')
        axs[0].plot([self.end_time, self.end_time],
                    [np.min([self.x, self.y, self.z]),
                     np.max([self.x, self.y, self.z])], label='end', c='m')
        axs[0].set_xlabel('Time (seconds)')
        axs[0].set_ylabel('Displacement')
        # axs[0].set_title(f'{self.n_missing} missing data')
        # axs[0].legend()

        axs[1].plot(self.time, self.x_vel, label='x', c='r')
        axs[1].plot(self.time, self.y_vel, label='y', c='g')
        axs[1].plot(self.time, self.z_vel, label='z', c='b')
        axs[1].plot([self.start_time, self.start_time],
                    [np.min([self.x_vel, self.y_vel, self.z_vel]),
                     np.max([self.x_vel, self.y_vel, self.z_vel])], label='start', c='c')
        axs[1].plot([self.end_time, self.end_time],
                    [np.min([self.x_vel, self.y_vel, self.z_vel]),
                     np.max([self.x_vel, self.y_vel, self.z_vel])], label='end', c='m')

        axs[1].set_xlabel('Time (seconds)')
        axs[1].set_ylabel('Velocity')

        plt.subplots_adjust(top=0.90, hspace=0.38, left=0.12, bottom=0.12)
        return fig, axs

    def display_missing_info(self):
        if self.contain_missing:
            print(f'This trial contains {self.n_missing} missing data points.\n'
                  f'Among them, there are {len(self.missing_segments_movement)} segments occurred during the movement!\n'
                  f'The size of the missing segments are: {self.n_missing_segments_movement}\n')
        else:
            print(f'This trial does not contain any missing data!')

    def format_results(self):
        return pd.DataFrame({
            'contain_movement': self.contain_movement,
            'fs': self.fs,
            'fc': self.fc,
            'rt': self.rt,
            'mt': self.mt,
            'movement_dist': np.linalg.norm(self.end_pos - self.start_pos),
            'peak_vel': self.peak_vel,
            'time_to_peak_vel': self.time_to_peak_vel,
            'time_after_peak_vel': self.time_after_peak_vel,
            'peak_acc': self.peak_acc,
            'time_to_peak_acc': self.time_to_peak_acc,
            'time_after_peak_acc': self.time_after_peak_acc,
        }, index=[0])

    def display_results(self):
        _results = self.format_results()

        for col in _results.columns:
            print(f'{col}: {_results[col].values[0]:.2f}')

    def demo_plots(self, fig=None, axs=None):
        if axs is None:
            fig, axs = plt.subplots(1, 1)
            # plt.tight_layout()

        axs.plot(self.time, self.x, label='x', linewidth=3, c='c')
        axs.plot(self.time, self.y, label='y', linewidth=3, c='m')
        axs.plot(self.time, self.z, label='z', linewidth=3, c='y')

        # axs.plot(self.time_original, self.x_original, label='x', linewidth=3, linestyle=':', c='k')
        # axs.plot(self.time_original, self.y_original, label='y', linewidth=3, linestyle=':', c='k')
        # axs.plot(self.time_original, self.z_original, label='z', linewidth=3, linestyle=':', c='k')
        # axs.scatter(self.time_original[self.ind_missing], self.x_original[self.ind_missing], color='k')

        axs.plot([self.start_time, self.start_time],
                 [np.min([self.x, self.y, self.z]),
                  np.max([self.x, self.y, self.z])],
                 label='Movement Start', color='g', linewidth=4, linestyle=':')
        axs.plot([self.end_time, self.end_time],
                 [np.min([self.x, self.y, self.z]),
                  np.max([self.x, self.y, self.z])],
                 label='Movement End', color='r', linewidth=4, linestyle=':')
        axs.set_xlabel('Time (s)', fontdict={'fontsize': 16})
        axs.set_ylabel('Displacement (mm)', fontdict={'fontsize': 16})
        axs.set_xticklabels(axs.get_xticklabels(), fontsize=12)
        axs.set_yticklabels(axs.get_yticklabels(), fontsize=12)
        # axs[0].set_title(f'Displacement, {self.n_missing} missing data')
        axs.legend()

        return fig, axs

    def debug_traj_3d(self, ax=None, color='k'):
        """ Create a debug plot that shows 3d trajectories"""
        if ax is None:
            plt.figure()
            ax = plt.axes(projection='3d')
        else:
            if ax.name != '3d':
                raise ValueError('The input MPL ax has to be a 3D axis! You can simply create an axis with the '
                                 'projection keyword: \n'
                                 'ax = plt.axes(projection=\'3d\')')

        # # ax.plot3D(self.x_smooth, self.y_smooth, self.z_smooth)
        # ax.scatter(self.x_smooth, self.y_smooth, self.z_smooth, c='k')
        # ax.scatter(self.x_smooth[0], self.y_smooth[0], self.z_smooth[0], c='r')  # beginning
        # ax.scatter(self.x_smooth[-1], self.y_smooth[-1], self.z_smooth[-1], c='g')  # end

        ax.plot3D(self.x, self.y, self.z, c=color)
        ax.scatter(self.x[0], self.y[0], self.z[0], c='r')  # beginning
        ax.scatter(self.x[-1], self.y[-1], self.z[-1], c='g')  # end

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        return ax

    def debug_traj_2d(self, x_axis, y_axis, var='', ax=None, center=True, full_output=False):
        """
        :param x_axis: the trajectory axis to be plotted on the x-axis of the 2d plot
        :param y_axis: the trajectory axis to be plotted on the y-axis of the 2d plot
        :param var: the variable to be plotted
        :param ax: matplotlib 2d axis object
        :param center: whether to center the data
        :param full_output: if True, will also return plot x and y coordinates
        :return: matplotlib 2d axis object
        """

        def check_axis(axis_name, _var):
            axis_name = axis_name.lower()

            if _var != '':
                _var = f'_{_var}'

            if (axis_name == 'x') | (axis_name == 'y') | (axis_name == 'z'):
                pos = getattr(self, f'{axis_name}{_var}')
            elif axis_name == 'xy':
                pos = np.sqrt((getattr(self, f'x{_var}') ** 2 +
                               getattr(self, f'y{_var}') ** 2))
            elif axis_name == 'xz':
                pos = np.sqrt((getattr(self, f'x{_var}') ** 2 +
                               getattr(self, f'z{_var}') ** 2))
            elif axis_name == 'yz':
                pos = np.sqrt((getattr(self, f'y{_var}') ** 2 +
                               getattr(self, f'z{_var}') ** 2))
            else:
                raise ValueError("Unrecognized trajectory axis name! Only support 1D or 2D axes!")
            return pos - pos[0] if center else pos

        plot_x = check_axis(x_axis, var)
        plot_y = check_axis(y_axis, var)
        line = ax.plot(plot_x, plot_y, alpha=.5)
        ax.scatter(plot_x[0], plot_y[0], c='r')
        ax.scatter(plot_x[-1], plot_y[-1], c='g')

        if full_output:
            return ax, line, plot_x, plot_y
        else:
            return ax, line

    @property
    def movement_plane_ax(self):
        return self._movement_plane

    @movement_plane_ax.setter
    def movement_plane_ax(self, value):
        if value == 'xy':
            self._movement_plane = [0, 1]
        elif value == 'xz':
            self._movement_plane = [0, 2]
        elif value == 'yz':
            self._movement_plane = [1, 2]
        else:
            raise ValueError('Invalid principal directions! Please use the following: '
                             'xy, xz, or yz')

    @property
    def primary_dir(self):
        return self._primary_dir

    @primary_dir.setter
    def primary_dir(self, value):
        if value == 'x':
            self._primary_dir = 0
        elif value == 'y':
            self._primary_dir = 1
        elif value == 'z':
            self._primary_dir = 2
        else:
            raise ValueError('Invalid primary directions! Please use the following: '
                             'x, y, or z')

        self.secondary_dir = list(set(self.movement_plane_ax) - {self.primary_dir})[0]

    @property
    def ground_dir(self):
        return self._ground_dir

    @ground_dir.setter
    def ground_dir(self, value):
        if value == 'x':
            self._ground_dir = 0
        elif value == 'y':
            self._ground_dir = 1
        elif value == 'z':
            self._ground_dir = 2
        else:
            raise ValueError('Invalid principal directions! Please use the following: '
                             'x, y, or z')

        ground_points = np.random.random((3, 3))
        ground_points[:, self._ground_dir] = 0
        self.ground_plane = Plane.best_fit(ground_points)
