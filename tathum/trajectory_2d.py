# from .coord import Coord
from tathum.coord import Coord

import typing
from collections import OrderedDict
import numpy as np
from enum import Enum




# %%
class Trajectory2D:
    x = Coord()
    y = Coord()

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 displacement_process: OrderedDict,
                 velocity_process: OrderedDict,
                 acceleration_process: OrderedDict,
                 time: typing.Optional[np.ndarray] = None,
                 fs: typing.Optional[float] = None,
                 fc: typing.Optional[float] = None,
                 ):
        self.x_original, self.y_original = x.copy(), y.copy()  # store a copy of the original data
        self.x, self.y = x, y
        self.n_frames = len(self.x)

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

