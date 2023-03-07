"""
A custom class to store coordinate data.

Written by X.M. Wang.

Wang, X.M., & Welsh, T.N. (2023). TAT-HUM: Trajectory Analysis Toolkit for Human Movements in Python.

"""

import numpy as np
import pandas as pd


class Coord:
    """
    This is a descriptor class that stores the coordinates of a SINGLE trajectory dimension.
    """

    def __set_name__(self, owner, name):
        self.name = f'_{name}'

    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        if value is None:
            setattr(instance, self.name, None)
        elif (type(value) is pd.Series) | (type(value) is pd.DataFrame) | (type(value) is np.ndarray):
            if type(value) is pd.Series:
                value = value.to_numpy(dtype=float)
            dim = value.shape

            if len(dim) == 1:
                setattr(instance, self.name, value)
            else:
                raise ValueError(f'The input coordinate array has to be 1D, instead it is {len(dim)}D!')
        else:
            raise ValueError('The input coordinates have to be either a Numpy array or Pandas Series! \n'
                             f'Instead, you used {type(value)}')

