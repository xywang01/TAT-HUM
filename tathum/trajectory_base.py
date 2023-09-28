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
                 spline_order: int = 3, n_spline_fit: int = 100,
                 ):
        self.vel_threshold = vel_threshold
        self.unit = unit
        self.n_dim = n_dim
        self.fs = fs
        self.fc = fc
        self.missing_data_value = missing_data_value
        self.spline_order = spline_order
        self.n_spline_fit = n_spline_fit
        self.movement_selection_method = movement_selection_method
        self.movement_selection_sign = movement_selection_sign
        self.start_time, self.end_time, self.movement_ind = np.nan, np.nan, np.nan
        self.start_pos, self.end_pos = np.nan, np.nan
        self.movement_displacement, self.movement_velocity = np.nan, np.nan
        self.contain_movement = False

    @property
    def n_frames(self):
        return self._n_frames

    @n_frames.setter
    def n_frames(self, value):
        self._n_frames = value

    @property
    def movement_velocity(self):
        """
        The velocity coordinate used to determine the movement boundaries.
        """
        return self._movement_velocity

    @movement_velocity.setter
    def movement_velocity(self, value):
        self._movement_velocity = value

    @property
    def movement_displacement(self):
        """
        The displacement coordinate used to determine the movement boundaries.
        """
        return self._movement_displacement

    @movement_displacement.setter
    def movement_displacement(self, value):
        self._movement_displacement = value

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

    @abstractmethod
    def find_start_and_end_pos(self, time_cutoff: float):
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

    def validate_movement(self):
        """
        Check to see if the trajectory contains valid movement
        """
        if np.isnan(self.start_time) | np.isnan(self.end_time):
            return False
        elif len(self.movement_ind) <= self.spline_order:  # when few data points are detected
            return False
        else:
            return True

    @staticmethod
    def find_preprocess_order(preprocess_var: str):
        if preprocess_var == 'displacement':
            return 1
        elif preprocess_var == 'velocity':
            return 2
        elif preprocess_var == 'acceleration':
            return 3
        else:
            raise ValueError('The preprocess variable has to be either displacement, velocity, or acceleration!')

    def display_results(self):
        if self.contain_movement:
            print(f'This trial does contain movement! \n'
                  f'Reaction time (RT) is {self.start_time:.2f}; Movement time (MT) is {self.end_time - self.start_time:.2f} s. \n'
                  f'Movement distance is {np.linalg.norm(self.end_pos - self.start_pos):.2f} mm. \n'
                  '----------------------------------------------------------------------------------------------------')
        else:
            print('This trial does not contain any detectable movements!')


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




