"""
A class for 2D trajectory analysis.

This class inherits from the base class TrajectoryBase. It implements the abstract methods in the base class, and
provides concrete implementations for the abstract methods in the base class.

Written by X.M. Wang.

Wang, X.M., & Welsh, T.N. (2023). TAT-HUM: Trajectory Analysis Toolkit for Human Movements in Python.
"""

from tathum.trajectory_base import TrajectoryBase
from tathum.functions import *
import typing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Trajectory2D(TrajectoryBase):

    N_DIM = 2  # 2D trajectory, needed for the parent class method validate_size()

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,

                 displacement_preprocess: tuple[Preprocesses, ...] = (Preprocesses.LOW_BUTTER, ),
                 velocity_preprocess: tuple[Preprocesses, ...] = (Preprocesses.CENT_DIFF, ),
                 acceleration_preprocess: tuple[Preprocesses, ...] = (Preprocesses.CENT_DIFF, ),

                 transform_end_points: tuple[np.ndarray, np.ndarray] = None,
                 transform_to: np.ndarray = None,

                 time: typing.Optional[np.ndarray] = None,

                 primary_dir: str = 'y',  # the primary direction of the movement

                 unit: str = 'mm',
                 missing_data_value: float = 0.,
                 fs: typing.Optional[int] = None, fc: typing.Optional[int] = None,
                 vel_threshold: float = 50.,

                 movement_selection_ax: str = 'xy',
                 movement_selection_method: str = 'length',
                 movement_selection_sign: str = 'positive',
                 custom_compute_movement_boundary: typing.Optional[callable] = None,

                 movement_pos_time_cutoff: float = 0.2,

                 spline_order: int = 3, n_spline_fit: int = 100,
                 ):

        self.x_original, self.y_original = x.copy(), y.copy()  # store a copy of the original data
        self.x, self.y = x, y
        self.x_vel, self.y_vel = np.array([]), np.array([])
        self.x_acc, self.y_acc = np.array([]), np.array([])
        self.n_frames = self.validate_size(n_dim=self.N_DIM)
        self.primary_dir = primary_dir

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

        # initialize the base class
        super().__init__(
            unit=unit,
            missing_data_value=missing_data_value,
            fs=fs, fc=fc,
            vel_threshold=vel_threshold,
            movement_selection_method=movement_selection_method, movement_selection_sign=movement_selection_sign,
            custom_compute_movement_boundary=custom_compute_movement_boundary,
            spline_order=spline_order, n_spline_fit=n_spline_fit, )

        # fill in missing data before performing spatial transformation (otherwise the missing data value would be
        # transformed as well)
        self.contain_missing, self.n_missing, self.ind_missing = self.missing_data()

        # if the cutoff frequency is not specified, then it will be computed automatically
        if self.fc is None:
            self.fc = find_optimal_cutoff_frequency(self.x, self.fs)

        # transform data if needed
        self.transform_end_points = transform_end_points
        if self.transform_end_points is not None:
            self.transform_to = transform_to if transform_to is not None else np.array([0, 1])
            start_pos, end_pos = self.transform_end_points[0], self.transform_end_points[1]
            self.transform_mat, self.transform_origin = self.compute_transform(start_pos, end_pos, self.transform_to)
            self.x, self.y = self.transform_data(self.x, self.y)
            self.x_original, self.y_original = self.transform_data(self.x_original, self.y_original)

        self.preprocess('displacement', displacement_preprocess)
        self.preprocess('velocity', velocity_preprocess)
        self.preprocess('acceleration', acceleration_preprocess)

        self.movement_displacement = self.find_movement_displacement(movement_selection_ax=movement_selection_ax)
        self.movement_velocity = self.find_movement_velocity(movement_selection_ax=movement_selection_ax)
        self.movement_acceleration = self.find_movement_acceleration(movement_selection_ax=movement_selection_ax)

        self.start_time, self.end_time, self.movement_ind = self.compute_movement_boundaries()
        self.contain_movement = self.validate_movement()

        if self.contain_movement:
            self.rt = self.start_time
            self.mt = self.end_time - self.start_time
            self.start_pos, self.end_pos = self.find_start_and_end_pos(time_cutoff=movement_pos_time_cutoff)

            self.time_movement = self.time[self.movement_ind]
            self.x_movement = self.x[self.movement_ind]
            self.y_movement = self.y[self.movement_ind]
            self.x_vel_movement = self.x_vel[self.movement_ind]
            self.y_vel_movement = self.y_vel[self.movement_ind]
            self.x_acc_movement = self.x_acc[self.movement_ind]
            self.y_acc_movement = self.y_acc[self.movement_ind]

            # trim the movement velocity to find the peak velocity
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

            self.time_fit, self.x_fit, self.x_spline = self.b_spline_fit_1d(self.time_movement, self.x_movement, self.n_spline_fit)
            _, self.y_fit, self.y_spline = self.b_spline_fit_1d(self.time_movement, self.y_movement, self.n_spline_fit)
            self.x_vel_fit = cent_diff(self.time_fit, self.x_fit)
            self.y_vel_fit = cent_diff(self.time_fit, self.y_fit)
            self.x_acc_fit = cent_diff(self.time_fit, self.x_vel_fit)
            self.y_acc_fit = cent_diff(self.time_fit, self.y_vel_fit)
        else:
            self.rt = None
            self.mt = None
            self.start_pos, self.end_pos = None, None

            self.time_movement = None
            self.x_movement = None
            self.y_movement = None
            self.x_vel_movement = None
            self.y_vel_movement = None
            self.x_acc_movement = None
            self.y_acc_movement = None

            self.vel_movement = None
            self.peak_vel = None
            self.time_to_peak_vel = None
            self.time_after_peak_vel = None

            self.movement_acceleration = None
            self.peak_acc = None
            self.time_to_peak_acc = None
            self.time_after_peak_acc = None

            self.time_fit = None
            self.x_fit = None
            self.x_spline = None
            self.y_fit = None
            self.y_spline = None
            self.x_vel_fit = None
            self.y_vel_fit = None
            self.x_acc_fit = None
            self.y_acc_fit = None

    def assign_preprocess_function(self,
                                   preprocess_var: str,
                                   preprocess: Preprocesses,):
        """
        A concrete implementation of the base class TrajectoryBase.

        Crucially, this assigns the unique preprocess wrapper functions unique to the current class. See the wrapper
        functions' respective implementations for details.
        """
        preprocess_order = self.find_preprocess_order(preprocess_var)
        if preprocess == Preprocesses.LOW_BUTTER:
            return self.low_butter, (preprocess_order, )
        elif preprocess == Preprocesses.CENT_DIFF:
            return self.cent_diff, (preprocess_order, )  # return a tuple of arguments
        else:
            raise ValueError('The preprocess is not recognized!')

    def low_butter(self, low_butter_order: int = 1):
        """
        A wrapper function for the low_butter() from tathum.functions specific to the current Trajectory2D class.
        """
        if low_butter_order == 1:
            self.x = low_butter(self.x, self.fs, self.fc)
            self.y = low_butter(self.y, self.fs, self.fc)
        elif low_butter_order == 2:
            self.x_vel = low_butter(self.x_vel, self.fs, self.fc)
            self.y_vel = low_butter(self.y_vel, self.fs, self.fc)
        elif low_butter_order == 3:
            self.x_acc = low_butter(self.x_acc, self.fs, self.fc)
            self.y_acc = low_butter(self.y_acc, self.fs, self.fc)
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

        elif cent_diff_order == 3:
            self.x_acc = cent_diff(self.time, self.x_vel)
            self.y_acc = cent_diff(self.time, self.y_vel)

        else:
            raise ValueError('The order of the central difference has to be either 1 (for velocity ) or 2 '
                             '(for acceleration!')

    def find_start_and_end_pos(self, time_cutoff: float = 0.2):
        """
        Implementation of the abstract method in the base class TrajectoryBase, specific to the current Trajectory2D
        class.
        """
        ind_start = (self.time > self.start_time - time_cutoff) & (self.time < self.start_time + time_cutoff)
        if np.any(ind_start):
            start_x = self.x[ind_start]
            start_y = self.y[ind_start]
            mean_start = np.nanmean(np.array([start_x, start_y]), axis=1)
        else:
            mean_start = np.empty((2,)) * np.nan

        ind_end = (self.time > self.end_time - time_cutoff) & (self.time < self.end_time + time_cutoff)
        if np.any(ind_end) > 0:
            end_x = self.x[ind_end]
            end_y = self.y[ind_end]
            mean_end = np.nanmean(np.array([end_x, end_y]), axis=1)
        else:
            mean_end = np.empty((2,)) * np.nan

        return mean_start, mean_end

    def transform_data(self, x, y):
        coord = np.concatenate([np.expand_dims(x, axis=0),
                                np.expand_dims(y, axis=0)], axis=0)
        coord_rot = np.matmul(self.transform_mat, coord - self.transform_origin)  # + self.transform_origin - leaving this out so that the trajectory is centered at the origin
        return coord_rot[0], coord_rot[1]

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

    def missing_data(self):
        """
        A wrapper function for the fill_missing_data() from tathum.functions.
        """
        self.x, self.y, self.time, missing_info = fill_missing_data(
            x=self.x, y=self.y, time=self.time, missing_data_value=self.missing_data_value,
        )
        self.n_frames = self.validate_size(self.N_DIM)  # remember to update n_frames
        return missing_info['contain_missing'], missing_info['n_missing'], missing_info['missing_ind']

    @staticmethod
    def compute_transform(start_pos, end_pos, transform_to):
        """
        A wrapper for the compute_transformation_2d() from tathum.functions.
        """
        return compute_transformation_2d(start_pos, end_pos, transform_to), np.expand_dims(start_pos, 1)  # use the start position as the origin

    @property
    def transform_end_points(self):
        """
        The 2D end points used to determine the transformation matrix.
        """
        return self._transform_end_point

    @transform_end_points.setter
    def transform_end_points(self, value):
        if value is None:
            self._transform_end_point = None
        else:
            err_msg = 'The transformation end point has to be a tuple of two 2D Numpy arrays! \n'
            if type(value) is not tuple:
                raise ValueError(f'{err_msg}'
                                 'The supplied value is not a tuple! ')
            elif len(value) != 2:
                raise ValueError(f'{err_msg}'
                                 f'The supplied tuple has {len(value)} elements instead of 2!')
            elif np.any([not type(v) == np.ndarray for v in value]):
                raise ValueError(f'{err_msg}'
                                 f'The supplied tuple contains elements that are not Numpy arrays!')
            elif np.any([not v.shape == (2, ) for v in value]):
                raise ValueError(f'{err_msg}'
                                 f'The supplied tuple contains elements are of size {[v.shape for v in value]}!')

            self._transform_end_point = value

    def find_movement_angle(self, perc_of_movement=0.2):
        """
        Find the angle of the movement based on the first x% of the movement on movement_plane_ax and primary_dir.
        :param perc_of_movement: the percentage of the movement to consider when computing the movement angle
        :return: the movement angle
        """
        if self.contain_movement:
            end_idx = int(np.round(perc_of_movement * len(self.x_movement)))

            displacement_vector = np.concatenate([np.expand_dims(self.x, axis=1),
                                                  np.expand_dims(self.y, axis=1)], axis=1)

            start_pos = displacement_vector[0]
            end_pos = displacement_vector[end_idx]
            movement_vector = end_pos - start_pos

            primary_dir_vector = np.zeros((3,))
            primary_dir_vector[self.primary_dir] = 1

            movement_angle = np.arccos(np.dot(movement_vector, primary_dir_vector) / np.linalg.norm(movement_vector))
            return movement_angle
        else:
            return None

    def find_movement_displacement(self, movement_selection_ax: str = 'xy'):
        if movement_selection_ax == 'x':
            return self.x
        elif movement_selection_ax == 'y':
            return self.y
        elif movement_selection_ax == 'xy':
            return np.linalg.norm(np.concatenate([
                np.expand_dims(self.x, axis=1),
                np.expand_dims(self.y, axis=1),
            ], axis=1), axis=1)
        else:
            raise ValueError('The movement selection axis has to be either x, y, or xy!')

    def find_movement_velocity(self, movement_selection_ax: str = 'xy'):
        if movement_selection_ax == 'x':
            return self.x_vel
        elif movement_selection_ax == 'y':
            return self.y_vel
        elif movement_selection_ax == 'xy':
            return np.linalg.norm(np.concatenate([
                np.expand_dims(self.x_vel, axis=1),
                np.expand_dims(self.y_vel, axis=1),
            ], axis=1), axis=1)
        else:
            raise ValueError('The movement selection axis has to be either x, y, or xy!')

    def find_movement_acceleration(self, movement_selection_ax: str = 'xy'):
        if movement_selection_ax == 'x':
            return self.x_acc
        elif movement_selection_ax == 'y':
            return self.y_acc
        elif movement_selection_ax == 'xy':
            return np.linalg.norm(np.concatenate([
                np.expand_dims(self.x_acc, axis=1),
                np.expand_dims(self.y_acc, axis=1),
            ], axis=1), axis=1)
        else:
            raise ValueError('The movement selection axis has to be either x, y, or xy!')

    def debug_plots(self, fig=None, axs=None):

        if axs is None:
            fig, axs = plt.subplots(2, 1)
            # plt.tight_layout()

        axs[0].plot(self.time_original, self.x_original, label='x', linestyle=':')
        axs[0].plot(self.time_original, self.y_original, label='y', linestyle=':')
        axs[0].scatter(self.time_original[self.ind_missing], self.x_original[self.ind_missing], color='k')

        axs[0].plot(self.time, self.x, label='x', c='r')
        axs[0].plot(self.time, self.y, label='y', c='g')

        # if self.contain_missing:
        #     for i_seg, seg in enumerate(self.missing_segments_movement):
        #         n_missing = self.n_missing_segments_movement[i_seg]
        #         seg_mid = int(np.median(seg) - 1)
        #
        #         axs[0].text(self.time[seg_mid], np.min([self.x_original, self.y_original, self.z_original]),
        #                     f'{n_missing}', fontsize=12, )

        # axs[0].plot(self.time, np.ones((len(self.time), 1)) * self.start_pos[principal_dir],
        #             c='c', linestyle=':', linewidth=3)
        # axs[0].plot(self.time, np.ones((len(self.time), 1)) * self.end_pos[principal_dir],
        #                 c='m', linestyle=':', linewidth=3)

        axs[0].plot([self.start_time, self.start_time],
                    [np.min([self.x, self.y]),
                     np.max([self.x, self.y])], label='start', c='c')
        axs[0].plot([self.end_time, self.end_time],
                    [np.min([self.x, self.y]),
                     np.max([self.x, self.y])], label='end', c='m')
        axs[0].set_xlabel('Time (seconds)')
        axs[0].set_ylabel('Displacement')
        # axs[0].set_title(f'{self.n_missing} missing data')
        # axs[0].legend()

        axs[1].plot(self.time, self.x_vel, label='x', c='r')
        axs[1].plot(self.time, self.y_vel, label='y', c='g')
        axs[1].plot([self.start_time, self.start_time],
                    [np.min([self.x_vel, self.y_vel]),
                     np.max([self.x_vel, self.y_vel])], label='start', c='c')
        axs[1].plot([self.end_time, self.end_time],
                    [np.min([self.x_vel, self.y_vel]),
                     np.max([self.x_vel, self.y_vel])], label='end', c='m')

        axs[1].set_xlabel('Time (seconds)')
        axs[1].set_ylabel('Velocity')

        plt.subplots_adjust(top=0.90, hspace=0.38, left=0.12, bottom=0.12)
        return fig, axs

