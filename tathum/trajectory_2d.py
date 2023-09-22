# from .coord import Coord
from tathum.coord import Coord
from tathum.trajectory_base import TrajectoryBase
from tathum.functions import *

import typing
from collections import OrderedDict
import numpy as np
from enum import Enum


class Trajectory2D(TrajectoryBase):

    N_DIM = 2  # 2D trajectory, needed for the parent class method validate_size()

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,

                 # displacement_preprocess: tuple[Preprocesses],
                 # velocity_preprocess: tuple[Preprocesses],
                 # acceleration_preprocess: tuple[Preprocesses],

                 transform_end_point: tuple[np.ndarray, np.ndarray] = None,
                 transform_to: np.ndarray = None,

                 time: typing.Optional[np.ndarray] = None,

                 primary_dir: str = 'y',  # the primary direction of the movement

                 unit: str = 'mm',
                 missing_data_value: float = 0.,
                 fs: typing.Optional[int] = None, fc: typing.Optional[int] = None,
                 vel_threshold: float = 50.,
                 movement_selection_method: str = 'length', movement_selection_sign: str = 'positive',
                 spline_order: int = 3, n_fit: int = 100,
                 ):

        self.x_original, self.y_original = x.copy(), y.copy()  # store a copy of the original data
        self.x, self.y = x, y
        self.n_frames = self.validate_size(n_dim=self.N_DIM)

        # self.displacement_process = [get_function(p) for p in displacement_preprocess]

        self.transform_end_point = transform_end_point
        if self.transform_end_point is not None:
            self.transform_to = transform_to if transform_to is not None else np.array([0, 1])
            start_pos, end_pos = self.transform_end_point[0], self.transform_end_point[1]
            self.transform_mat, self.transform_origin = self.compute_transform(start_pos, end_pos, self.transform_to)

        self.primary_dir = primary_dir

        if (time is None) & (fs is None):
            raise ValueError('You have to either specify the time stamps or the sampling frequency!')

        if time is None:
            self.fs = fs
            self.time = np.linspace(0, self.n_frames * 1 / self.fs, num=self.n_frames, endpoint=False)
            self.time_original = self.time.copy()
        else:
            self.time_original = time.copy()
            self.time = time
            self.fs = 1 / np.mean(np.diff(self.time))
            if len(self.time) != self.n_frames:
                raise ValueError('The size of the input time stamps is not the same as the size of the coordinates!')

        super().__init__(
            unit=unit,
            missing_data_value=missing_data_value,
            fs=fs, fc=fc,
            vel_threshold=vel_threshold,
            movement_selection_method=movement_selection_method, movement_selection_sign=movement_selection_sign,
            spline_order=spline_order, n_fit=n_fit,)

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
        print(missing_info)
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

    def preprocess_displacement(self):
        pass

    def preprocess_velocity(self):
        pass

    def preprocess_acceleration(self):
        pass

    @property
    def movement_displacement(self):
        return self._movement_displacement

    @movement_displacement.setter
    def movement_displacement(self, value):
        self._movement_displacement = value

    @property
    def movement_velocity(self):
        return self._movement_velocity

    @movement_velocity.setter
    def movement_velocity(self, value):
        self._movement_velocity = value

    def find_start_and_end_pos(self):
        pass


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
        transform_end_point = temp_transform[['x', 'y']].to_numpy()
        transform_end_point = (transform_end_point[0], transform_end_point[1])  # reformat the data for the appropriate input type

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
                    time=temp['traj_ind'].values,

                    transform_end_point=transform_end_point,
                    transform_to=np.array([0, 1]),
                )

                plt.figure()
                plt.scatter(traj.x, traj.y)
                traj.missing_data()
                plt.plot(traj.x, traj.y)



                # plt.plot(traj.transform_origin[0], traj.transform_origin[1], marker='o', color='black')
                # x_rot, y_rot = traj.transform_data(traj.x, traj.y)
                # plt.plot(x_rot, y_rot)
                plt.axis('equal')
                raise ValueError



