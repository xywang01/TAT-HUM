from functools import reduce
from collections import OrderedDict
from enum import Enum
from tathum.functions import *


# DisplacementProcess = Enum('DisplacementProcess',
#                            ['missing_data', 'spatial_transform', 'butter_smooth', 'cent_diff'])
# VelocityProcess = Enum('VelocityProcess',
#                        ['butter_smooth', 'cent_diff'])
# AccelerationProcess = Enum('AccelerationProcess',
#                            ['butter_smooth', 'cent_diff'])

class TATHUMPreprocesses(Enum):
    FILL_MISSING = 1
    SPATIAL_TRANSFORM = 2
    LOW_BUTTER = 3
    DUAL_BUTTER = 4


def composite_function(*functions):
    def compose(f, g):
        return lambda x: f(g(x))

    return reduce(compose, functions, lambda x: x)


def get_function(process):
    if process == TATHUMPreprocesses.FILL_MISSING:
        return fill_missing_data
    elif process == TATHUMPreprocesses.SPATIAL_TRANSFORM:
        return composite_function(compute_transformation, rotate_coord)
    elif process == TATHUMPreprocesses.LOW_BUTTER:
        return low_butter
    # elif process == TATHUMProcesses.DUAL_BUTTER:
    #     return dual_butter


class TrajectoryBase:
    def __init__(self,
                 displacement_process: list[TATHUMPreprocesses],
                 # velocity_process: list[TATHUMPreprocesses],
                 # acceleration_process: list[TATHUMPreprocesses],
                 ):
        self.displacement_process = composite_function(*[get_function(p) for p in displacement_process])


base = TrajectoryBase([TATHUMPreprocesses.FILL_MISSING, TATHUMPreprocesses.SPATIAL_TRANSFORM,])

