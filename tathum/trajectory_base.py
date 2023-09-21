from abc import ABC, abstractmethod
from functools import reduce
from collections import OrderedDict
from enum import Enum

# from .coord import Coord
# from .functions import *
from tathum.coord import Coord
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
    CENT_DIFF = 4


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
    elif process == TATHUMPreprocesses.CENT_DIFF:
        return cent_diff


class TrajectoryBase(ABC):
    # pre-define the coordinates to be used for all concrete classes
    x = Coord()
    y = Coord()
    z = Coord()
    time = Coord()

    def __init__(self,

                 primary_dir: str = 'z',

                 unit: str = 'mm',
                 n_dim: int = 3,
                 missing_data_filler: float = 0.,
                 fs: int = 250, fc: int = 10,
                 vel_threshold: float = 50.,
                 movement_selection_method: str = 'length', movement_selection_sign: str = 'positive',
                 spline_order: int = 3, n_fit: int = 100,



                 displacement_process: tuple[TATHUMPreprocesses] = (TATHUMPreprocesses.FILL_MISSING,
                                                                    TATHUMPreprocesses.SPATIAL_TRANSFORM,
                                                                    TATHUMPreprocesses.LOW_BUTTER, ),
                 velocity_process: tuple[TATHUMPreprocesses] = (TATHUMPreprocesses.CENT_DIFF, ),
                 acceleration_process: tuple[TATHUMPreprocesses] = (TATHUMPreprocesses.CENT_DIFF, ),
                 ):
        self.vel_threshold = vel_threshold
        self.unit = unit
        self.n_dim = n_dim
        self.fs = fs
        self.fc = fc
        self.missing_data_filler = missing_data_filler
        self.spline_order = spline_order
        self.n_frames_fit = n_fit
        self.movement_selection_method = movement_selection_method
        self.movement_selection_sign = movement_selection_sign

        self.displacement_process = composite_function(*[get_function(p) for p in displacement_process])
        self.velocity_process = composite_function(*[get_function(p) for p in velocity_process])
        self.acceleration_process = composite_function(*[get_function(p) for p in acceleration_process])

    @property
    @abstractmethod
    def n_frames(self):
        """
        Number of frames in the trajectory.
        """
        pass

    @n_frames.setter
    @abstractmethod
    def n_frames(self, value):
        pass

    @property
    @abstractmethod
    def movement_velocity(self):
        """
        Class property used in compute_movement_boundaries().

        Derives the velocity, either along a single axis or the resultant velocity, that will be used to determine
        the movement boundaries. This need to be implemented in the concrete class depending on the trajectory's
        dimensions (2D or 3D).
        """
        pass

    @property
    @abstractmethod
    def movement_displacement(self):
        """
        Class property used in compute_movement_boundaries().

        Derives the displacement, either along a single axis or the resultant displacement, that will be used to
        automatically distinguish valid from invalid movements. This need to be implemented in the concrete class.
        """
        pass

    @abstractmethod
    def find_start_and_end_pos(self):
        """
        Abstract method to be called after compute_movement_boundaries().

        Find the start and end positions of the movement.
        """
        pass

    @staticmethod
    def b_spline_fit_1d(time_vec, coord, n_fit, smooth=0., return_spline=True):
        """
        A static method to fit a B-spline to a 1D coordinate.

        This is simply an interface to the function of the same name in tathum.functions.
        """
        return b_spline_fit_1d(time_vec, coord, n_fit, smooth=smooth, return_spline=return_spline)

    def validate_size(self, n_dim: int = 3):
        """ Validate input coordinate size. """
        n_x, n_y, n_z = len(self.x), len(self.y), len(self.z)
        if (n_dim == 3) & (not (n_x == n_y == n_z)):
            raise ValueError("The input x, y, and z have to be of the same size! \n"
                             f"Instead, len(x)={len(self.x)}, len(y)={len(self.y)}, len(z)={len(self.z)}")
        elif (n_dim == 2) & (not (n_x == n_y)):
            raise ValueError("The input x and y have to be of the same size! \n"
                             f"Instead, len(x)={len(self.x)}, len(y)={len(self.y)}")
        return n_x

    def compute_movement_boundaries(self):
        """
        Computes the movement boundaries based on the velocity profile.
        """
        vel_threshold_ind = np.where(self.movement_velocity >= self.vel_threshold)[0]

        if len(vel_threshold_ind) == 0:
            # in case there's no movement detected
            return np.nan, np.nan, np.nan

        vel_ind = consecutive(vel_threshold_ind)

        # in case there are multiple crossings at the threshold velocity
        if len(vel_ind) > 1:
            if self.movement_selection_method == 'length':
                # only use the portion of movement with the largest number of samples
                vel_len = [len(vel) for vel in vel_ind]
                max_vel = np.where(vel_len == np.max(vel_len))[0][0]
                vel_threshold_ind = vel_ind[max_vel]

            elif self.movement_selection_method == 'sign':
                movement_dist = []
                for vel in vel_ind:
                    movement_dist.append(self.movement_displacement[vel[-1]] - self.movement_displacement[vel[0]])
                movement_dist = np.array(movement_dist)

                if self.movement_selection_sign == 'positive':
                    segment_id = np.where(movement_dist > 0)[0]
                else:
                    segment_id = np.where(movement_dist < 0)[0]

                if len(segment_id) == 1:
                    vel_threshold_ind = vel_ind[segment_id[0]]
                elif len(segment_id) > 1:
                    # get the segment with the most data
                    vel_len = [len(vel_ind[seg_id]) for seg_id in segment_id]
                    max_vel = np.where(vel_len == np.max(vel_len))[0][0]
                    vel_threshold_ind = vel_ind[segment_id[max_vel]]
                else:
                    print('no valid movement detected!')
                    return np.nan, np.nan, np.nan

        move_start_ind = vel_threshold_ind[0] - 1 if vel_threshold_ind[0] > 0 else 0
        move_end_ind = vel_threshold_ind[-1] + 1 if vel_threshold_ind[-1] < self.n_frames - 1 else self.n_frames - 1

        time_start = self.time[move_start_ind]
        time_end = self.time[move_end_ind]

        return time_start, time_end, vel_threshold_ind







#
# class TrajectoryTest(TrajectoryBase):
#     def __init__(self, x, y, z):
#         super().__init__()
#         self.x = x
#         self.y = y
#         self.z = z
#
#         self.n_frames = self.validate_size()
#
#     @property
#     def n_frames(self):
#         return self._n_frames
#
#     @n_frames.setter
#     def n_frames(self, value):
#         self._n_frames = value
#
#     def movement_displacement(self):
#         return self.x
#
#     def movement_velocity(self):
#         return self.x
#
#     def compute_movement_boundaries(self):
#         super().compute_movement_boundaries()
#
#
# test = TrajectoryTest(x=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ),
#                       y=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ),
#                       z=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ))
#
# # test.compute_movement_boundaries()
# print(test.n_frames)




