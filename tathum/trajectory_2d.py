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
                 time: typing.Optional[np.ndarray] = None,

                 primary_dir: str = 'y',  # the primary direction of the movement


                 displacement_process: tuple[Preprocesses] = (Preprocesses.FILL_MISSING,
                                                              Preprocesses.SPATIAL_TRANSFORM,
                                                              Preprocesses.LOW_BUTTER,),
                 velocity_process: tuple[Preprocesses] = (Preprocesses.CENT_DIFF,),
                 acceleration_process: tuple[Preprocesses] = (Preprocesses.CENT_DIFF,),

                 unit: str = 'mm',
                 missing_data_filler: float = 0.,
                 fs: typing.Optional[int] = None, fc: typing.Optional[int] = None,
                 vel_threshold: float = 50.,
                 movement_selection_method: str = 'length', movement_selection_sign: str = 'positive',
                 spline_order: int = 3, n_fit: int = 100,
                 ):

        self.x_original, self.y_original = x.copy(), y.copy()  # store a copy of the original data
        self.x, self.y = x, y
        self.n_frames = self.validate_size(n_dim=self.N_DIM)

        self.displacement_process = [get_function(p) for p in displacement_process]

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
            missing_data_filler=missing_data_filler,
            fs=fs, fc=fc,
            vel_threshold=vel_threshold,
            movement_selection_method=movement_selection_method, movement_selection_sign=movement_selection_sign,
            spline_order=spline_order, n_fit=n_fit,)

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






raw_data = np.genfromtxt('./demo/demo_data/demo_data_2d.csv', delimiter=',')

traj_test = Trajectory2D(x=raw_data[:, 0], y=raw_data[:, 1], time=raw_data[:, 2])


