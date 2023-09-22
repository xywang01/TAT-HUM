from abc import ABC, abstractmethod

# from .coord import Coord
# from .functions import *
from tathum.coord import Coord
from tathum.functions import *


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
                 missing_data_value: float = 0.,
                 fs: int = 250, fc: int = 10,
                 vel_threshold: float = 50.,
                 movement_selection_method: str = 'length', movement_selection_sign: str = 'positive',
                 spline_order: int = 3, n_fit: int = 100,
                 ):
        self.vel_threshold = vel_threshold
        self.unit = unit
        self.n_dim = n_dim
        self.fs = fs
        self.fc = fc
        self.missing_data_value = missing_data_value
        self.spline_order = spline_order
        self.n_frames_fit = n_fit
        self.movement_selection_method = movement_selection_method
        self.movement_selection_sign = movement_selection_sign

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

    def preprocess(self, preprocess_var: str, preprocess_tuple: tuple):
        """
        A generic method that goes through a tuple of preprocesses and assign the corresponding function and relevant
        input.
        :param preprocess_var: The variable to be preprocessed
        :param preprocess_tuple: The tuple of preprocesses
        """
        for p in preprocess_tuple:
            f, val = self.assign_preprocess_function(preprocess_var, p)
            f() if val is None else f(*val)

    @abstractmethod
    def assign_preprocess_function(self, preprocess_var: str, preprocess: Preprocesses) -> (callable, tuple):
        """
        An abstract method that all concrete classes need to implement. This method assigns the preprocess function
        unique to the concrete class and the relevant variable to be preprocessed.
        :param preprocess_var: The variable to be preprocessed
        :param preprocess: The preprocess to be applied
        :return: The preprocess function and the relevant input
        """
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

    @movement_velocity.setter
    @abstractmethod
    def movement_velocity(self, value):
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

    @movement_displacement.setter
    @abstractmethod
    def movement_displacement(self, value):
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
        n_x, n_y = len(self.x), len(self.y)
        if n_dim == 3:
            n_z = len(self.z)
            if not (n_x == n_y == n_z):
                raise ValueError("The input x, y, and z have to be of the same size! \n"
                                 f"Instead, len(x)={len(self.x)}, len(y)={len(self.y)}, len(z)={len(self.z)}")
        elif n_dim == 2:
            if not (n_x == n_y):
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




