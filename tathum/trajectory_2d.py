# from .coord import Coord
from tathum.trajectory_base import TrajectoryBase
from tathum.functions import *
import typing
import numpy as np


class Trajectory2D(TrajectoryBase):

    N_DIM = 2  # 2D trajectory, needed for the parent class method validate_size()

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,

                 displacement_preprocess: tuple[Preprocesses, ...] = (Preprocesses.LOW_BUTTER, ),
                 velocity_preprocess: tuple[Preprocesses, ...] = (Preprocesses.CENT_DIFF, ),
                 acceleration_preprocess: tuple[Preprocesses, ...] = (Preprocesses.CENT_DIFF, ),

                 transform_end_point: tuple[np.ndarray, np.ndarray] = None,
                 transform_to: np.ndarray = None,

                 time: typing.Optional[np.ndarray] = None,

                 primary_dir: str = 'y',  # the primary direction of the movement

                 unit: str = 'mm',
                 missing_data_value: float = 0.,
                 fs: typing.Optional[int] = None, fc: typing.Optional[int] = None,
                 vel_threshold: float = 50.,

                 movement_selection_ax: str = 'y',
                 movement_selection_method: str = 'length', movement_selection_sign: str = 'positive',

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

        super().__init__(
            unit=unit,
            missing_data_value=missing_data_value,
            fs=fs, fc=fc,
            vel_threshold=vel_threshold,
            movement_selection_method=movement_selection_method, movement_selection_sign=movement_selection_sign,
            spline_order=spline_order, n_spline_fit=n_spline_fit, )

        # fill in missing data before performing spatial transformation (otherwise the missing data value would be
        # transformed as well)
        self.contain_missing, self.n_missing, self.ind_missing = self.missing_data()

        # if the cutoff frequency is not specified, then it will be computed automatically
        if self.fc is None:
            self.fc = find_optimal_cutoff_frequency(self.x, self.fs)

        # transform data if needed
        self.transform_end_point = transform_end_point
        if self.transform_end_point is not None:
            self.transform_to = transform_to if transform_to is not None else np.array([0, 1])
            start_pos, end_pos = self.transform_end_point[0], self.transform_end_point[1]
            self.transform_mat, self.transform_origin = self.compute_transform(start_pos, end_pos, self.transform_to)
            self.x, self.y = self.transform_data(self.x, self.y)

        self.preprocess('displacement', displacement_preprocess)
        self.preprocess('velocity', velocity_preprocess)
        self.preprocess('acceleration', acceleration_preprocess)

        self.movement_displacement = self.find_movement_displacement(movement_selection_ax=movement_selection_ax)
        self.movement_velocity = self.find_movement_velocity(movement_selection_ax=movement_selection_ax)

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

            self.time_fit, self.x_fit, self.x_spline = self.b_spline_fit_1d(self.time_movement, self.x_movement, self.n_spline_fit)
            _, self.y_fit, self.y_spline = self.b_spline_fit_1d(self.time_movement, self.y_movement, self.n_spline_fit)
            _, self.x_vel_fit, self.x_vel_spline = self.b_spline_fit_1d(self.time_movement, self.x_vel_movement, self.n_spline_fit)
            _, self.y_vel_fit, self.y_vel_spline = self.b_spline_fit_1d(self.time_movement, self.y_vel_movement, self.n_spline_fit)
            _, self.x_acc_fit, self.x_acc_spline = self.b_spline_fit_1d(self.time_movement, self.x_acc_movement, self.n_spline_fit)
            _, self.y_acc_fit, self.y_acc_spline = self.b_spline_fit_1d(self.time_movement, self.y_acc_movement, self.n_spline_fit)
        else:
            self.rt = np.nan
            self.mt = np.nan
            self.start_pos, self.end_pos = np.empty(3) * np.nan, np.empty(3) * np.nan

            self.time_movement = np.nan
            self.x_movement = np.nan
            self.y_movement = np.nan
            self.x_vel_movement = np.nan
            self.y_vel_movement = np.nan
            self.x_acc_movement = np.nan
            self.y_acc_movement = np.nan

            self.x_spline = np.nan
            self.y_spline = np.nan
            self.x_vel_spline = np.nan
            self.y_vel_spline = np.nan
            self.x_acc_spline = np.nan
            self.y_acc_spline = np.nan

    def find_start_and_end_pos(self, time_cutoff: float = 0.2):
        ind_start = (self.time < self.start_time) & (self.time > self.start_time - time_cutoff)
        if np.any(ind_start):
            start_x = self.x[ind_start]
            start_y = self.y[ind_start]
            mean_start = np.mean(np.array([start_x, start_y]), axis=1)
        else:
            mean_start = np.empty((2,)) * np.nan

        ind_end = (self.time > self.end_time) & (self.time < self.end_time + time_cutoff)
        if np.any(ind_end) > 0:
            end_x = self.x[ind_end]
            end_y = self.y[ind_end]
            mean_end = np.mean(np.array([end_x, end_y]), axis=1)
        else:
            mean_end = np.empty((2,)) * np.nan

        return mean_start, mean_end

    def assign_preprocess_function(self,
                                   preprocess_var: str,
                                   preprocess: Preprocesses,):
        if preprocess_var == 'displacement':
            preprocess_order = 1
        elif preprocess_var == 'velocity':
            preprocess_order = 2
        elif preprocess_var == 'acceleration':
            preprocess_order = 3
        else:
            raise ValueError('The preprocess variable has to be either displacement, velocity, or acceleration!')

        if preprocess == Preprocesses.LOW_BUTTER:
            return self.low_butter, (preprocess_order, )
        elif preprocess == Preprocesses.CENT_DIFF:
            return self.cent_diff, (preprocess_order, )  # return a tuple of arguments

    def low_butter(self, low_butter_order: int = 1):
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
        if cent_diff_order == 2:
            self.x_vel = cent_diff(self.time, self.x)
            self.y_vel = cent_diff(self.time, self.y)

        elif cent_diff_order == 3:
            self.x_acc = cent_diff(self.time, self.x_vel)
            self.y_acc = cent_diff(self.time, self.y_vel)

        else:
            raise ValueError('The order of the central difference has to be either 1 (for velocity ) or 2 '
                             '(for acceleration!')

    def transform_data(self, x, y):
        coord = np.concatenate([np.expand_dims(x, axis=0),
                                np.expand_dims(y, axis=0)], axis=0)
        coord_rot = np.matmul(self.transform_mat, coord - self.transform_origin)  # + self.transform_origin - leaving this out so that the trajectory is centered at the origin
        return coord_rot[0], coord_rot[1]

    def missing_data(self):
        """
        A wrapper function for the fill_missing_data() from tathum.functions.
        """

        self.x, self.y, self.time, missing_info = fill_missing_data(
            x=self.x, y=self.y, time=self.time, missing_data_value=self.missing_data_value,
        )
        return missing_info['contain_missing'], missing_info['n_missing'], missing_info['missing_ind']

    @staticmethod
    def compute_transform(start_pos, end_pos, transform_to):
        """
        A wrapper for the compute_transformation_2d() from tathum.functions.
        """
        return compute_transformation_2d(start_pos, end_pos, transform_to), np.expand_dims(start_pos, 1)  # use the start position as the origin

    @property
    def transform_end_point(self):
        return self._transform_end_point

    @transform_end_point.setter
    def transform_end_point(self, value):
        if value is None:
            self._transform_end_point = None
        else:
            if type(value) is not tuple:
                raise ValueError('The transformation end point has to be a tuple of two 2D Numpy arrays! \n'
                                 'The supplied value is not a tuple! ')
            elif len(value) != 2:
                raise ValueError('The transformation end point has to be a tuple of two 2D Numpy arrays! \n'
                                 f'The supplied tuple has {len(value)} elements instead of 2!')
            elif np.any([not type(v) == np.ndarray for v in value]):
                raise ValueError('The transformation end point has to be a tuple of two 2D Numpy arrays! \n'
                                 f'The supplied tuple contains elements that are not Numpy arrays!')
            elif np.any([not v.shape == (2, ) for v in value]):
                raise ValueError('The transformation end point has to be a tuple of two 2D Numpy arrays! \n'
                                 f'The supplied tuple contains elements are of size {[v.shape for v in value]}!')

            self._transform_end_point = value

    @property
    def n_frames(self):
        return self._n_frames

    @n_frames.setter
    def n_frames(self, value):
        self._n_frames = value

    @property
    def movement_displacement(self):
        """
        The displacement coordinate used to determine the movement boundaries.
        """
        return self._movement_displacement

    @movement_displacement.setter
    def movement_displacement(self, value):
        self._movement_displacement = value

    def find_movement_displacement(self, movement_selection_ax: str = 'y'):
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

    @property
    def movement_velocity(self):
        return self._movement_velocity

    @movement_velocity.setter
    def movement_velocity(self, value):
        self._movement_velocity = value

    def find_movement_velocity(self, movement_selection_ax: str = 'y'):
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


import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('./demo/demo_data/demo_data_2d.csv')
transform_data = pd.read_csv('./demo/demo_data/transform_data_2d.csv')

par_id_all = raw_data['par_id'].unique()
vib_all = raw_data['vib'].unique()
modality_all = raw_data['modality'].unique()
vision_all = raw_data['vision'].unique()

for par_id in par_id_all:
    for modality in modality_all:
        temp_transform = transform_data[
            (transform_data['par_id'] == par_id) &
            (transform_data['modality'] == modality)]
        exp_end_point = temp_transform[['x', 'y']].to_numpy()
        exp_end_point = (exp_end_point[0], exp_end_point[1])  # reformat the data for the appropriate input type

        for vib in vib_all:
            for vision in vision_all:
                df_idx = np.where((raw_data['par_id'] == par_id) &
                                  (raw_data['vib'] == vib) &
                                  (raw_data['modality'] == modality) &
                                  (raw_data['vision'] == vision))[0]
                temp = raw_data.iloc[df_idx]

                temp_x = temp['x'].values
                temp_y = temp['y'].values

                temp_x[[2, 3, 4, 6, 7]] = 0.
                temp_y[[2, 3, 4, 6, 7]] = 0.

                traj = Trajectory2D(
                    x=temp_x,
                    y=temp_y,
                    fs=60,  # Hz, time stamp is not available from the dataset

                    transform_end_point=exp_end_point,
                    transform_to=np.array([0, 1]),

                    displacement_preprocess=(Preprocesses.LOW_BUTTER, ),
                    velocity_preprocess=(Preprocesses.CENT_DIFF, Preprocesses.LOW_BUTTER, ),
                    acceleration_preprocess=(Preprocesses.CENT_DIFF, Preprocesses.LOW_BUTTER, ),
                )

                plt.figure()

                plt.scatter(traj.x, traj.y)
                plt.plot(traj.x_fit, traj.y_fit)

                # plt.plot(traj.transform_origin[0], traj.transform_origin[1], marker='o', color='black')
                # x_rot, y_rot = traj.transform_data(traj.x, traj.y)
                # plt.plot(x_rot, y_rot)
                plt.axis('equal')
                raise ValueError



