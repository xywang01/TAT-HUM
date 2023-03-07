"""
The Trajectory class that takes the raw trajectory data and automatically processes the data to derive relevant action
measures.

Written by X.M. Wang.

Wang, X.M., & Welsh, T.N. (2023). TAT-HUM: Trajectory Analysis Toolkit for Human Movements in Python.

"""

from .coord import Coord

import numpy as np
from scipy import interpolate
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation
from skspatial.objects import Plane, Points, Vector
from pytransform3d.rotations import matrix_from_axis_angle
import matplotlib.pyplot as plt


class Trajectory:
    """
    Automatically processes trajectory data.
    """
    x = Coord()
    y = Coord()
    z = Coord()
    time = Coord()

    def __init__(self,
                 x, y, z, time=None,
                 transform_end_points=None,  # end points used for spatial transformation
                 principal_dir='xz',  # 2D plane that specifies the principal direction of the reach
                 primary_dir='z',  # the primary movement direction
                 ground_dir='y',  # the ground's normal direction
                 vel_threshold=50.,
                 time_cutoff=0.2,  # used for finding the start and end positions
                 n_fit=100, fs=250, fc=10, spline_order=3,
                 unit='mm',
                 missing_data_filler=0.,
                 movement_selection_method='length',
                 movement_selection_sign=None,
                 movement_selection_dir=None,
    ):
        """
        :param n_fit: number of fitted time stamps for the B-spline fit.
        :param time: the time stamps, if not supplied will calculate using sampling frequency.
        :param fs: sampling frequency.
        :param fc: cutoff frequency for the low-pass Butterworth filter.
        :param unit: position measurement unit.
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        :param time: the corresponding time stamps
        :param transform_end_points: A minimum of three 3D points that specifies the plane on which the movement
        occurred. If provided, the best-fitting plane for these points will be computed and used to transform the
        trajectory so that the transformed movement would take place on a horizontal surface. This could be derived
        using the same Trajectory class.
        :param principal_dir: The principal directions of the movement.
        :param vel_threshold: The velocity threshold to determine movement initiation and termination.
        :param time_cutoff: The amount of time before/after the movement initiation/termination to consider when
        computing the trajectory's end positions. Instead of using the position at a particular time, the end position
        is the average of all positions within this time cutoff range.
        :param mt_threshold: threshold for determining whether the trial contains valid movement. MT below this
        threshold would be considered invalid
        :param n_fit: number of fitted trajectory points when using bspline
        :param fs: sampling frequency
        :param fc: cutoff frequency (for the Butterworth filter)
        :param spline_order: the order of the b-spline fit, default to 3
        :param unit: position measure's unit
        :param missing_data_filler: the filler value for missing data, default to be 0.
        :param movement_selection_method: the method to select the movement segment in case extraneous movements are
        detected. Can be chosen from ('length', 'sign', ). 'length' simply select the movement segment with the most
        number of points, whereas 'sign' selects the movement segment based on movement direction. 'length' is more
        effective for small extraneous movements whereas 'sign' is more effective for movement reversals during the data
        collection period. See movement_selection_sign for more.
        :param movement_selection_sign: can be chosen from ('positive', 'negative'), only applies to directional
        movement. The sign is determined by the difference between the end and start position.
        :param movement_selection_dir: the axis ('x', 'y', 'z') along which the movement selection should be performed
        """
        self.unit = unit
        self.fs = fs
        self.fc = fc
        self.missing_data_filler = missing_data_filler
        self.spline_order = spline_order

        self.x_original, self.y_original, self.z_original = x.copy(), y.copy(), z.copy()  # keep a separate copy of the original data
        self.x, self.y, self.z = x, y, z
        self.n_frames = self.validate_size()
        self.principal_dir = principal_dir
        self.secondary_dir = None
        self.primary_dir = primary_dir
        self.ground_plane = None
        self.ground_dir = ground_dir

        self.movement_selection_method = movement_selection_method
        self.movement_selection_sign = movement_selection_sign
        if self.movement_selection_method == 'sign' :
            if self.movement_selection_sign is None:
                raise ValueError('movement_selection_method is set to "sign" but movement_selection_sign was not specified!')
            if movement_selection_dir is None:
                self.movement_selection_dir = self.primary_dir
            else:
                if movement_selection_dir == 'x':
                    self.movement_selection_dir = 0
                elif movement_selection_dir == 'y':
                    self._primary_dir = 1
                elif movement_selection_dir == 'z':
                    self._primary_dir = 2
                else:
                    raise ValueError('Invalid movement_selection_dir! Please use the following: x, y, or z')

        self.contain_movement = True  # whether there was actual movement

        # set the time stamps. If time stamps are not supplied, generate evenly spaced time stamps based on sampling
        # frequency and number of frames.
        if time is not None:
            self.time_original = time
            self.time = time
            if len(self.time) != self.n_frames:
                raise ValueError('The size of the input time stamps is not the same as the size of the coordinates!')
        else:
            self.time = np.linspace(0, self.n_frames * 1 / self.fs, num=self.n_frames, endpoint=False)
            self.time_original = self.time.copy()

        # eliminate missing data; need to do it before the transformation
        self.contain_missing, self.n_missing, self.ind_missing = self.missing_data()

        # transform the data if necessary
        if transform_end_points is not None:
            self.x, self.y, self.z = self.transform_data(self.x, self.y, self.z, transform_end_points)
            self.x_original, self.y_original, self.z_original = self.transform_data(
                self.x_original, self.y_original, self.z_original, transform_end_points)

        # smooth position
        self.x_smooth = self.low_butter_scipy(self.x)
        self.y_smooth = self.low_butter_scipy(self.y)
        self.z_smooth = self.low_butter_scipy(self.z)

        # # parameterize and fit the trajectory
        self.n_frames_fit = n_fit

        # 3d distance to the origin
        self.pos_3d = np.sqrt(self.x_smooth ** 2 + self.y_smooth ** 2 + self.z_smooth ** 2)
        self.vel_3d = self.cent_diff(self.time, self.pos_3d)

        # compute velocity
        self.x_vel = self.cent_diff(self.time, self.x_smooth)
        self.y_vel = self.cent_diff(self.time, self.y_smooth)
        self.z_vel = self.cent_diff(self.time, self.z_smooth)

        # smooth velocity
        self.x_vel_smooth = self.low_butter_scipy(self.x_vel)
        self.y_vel_smooth = self.low_butter_scipy(self.y_vel)
        self.z_vel_smooth = self.low_butter_scipy(self.z_vel)

        # identify movement initiation and termination
        self.start_time, self.end_time, self.movement_ind = self.find_time(vel_threshold)
        if np.isnan(self.start_time) | np.isnan(self.end_time):
            self.contain_movement = False
        elif len(self.movement_ind) <= self.spline_order:  # when few data points are detected
            self.contain_movement = False

        if self.contain_movement:
            self.rt = self.start_time
            self.mt = self.end_time - self.start_time

            self.start_pos, self.end_pos = self.find_start_and_end_pos(time_cutoff)

            # normalize the start and end positions so that the reach starts at the origin
            self.start_pos_norm = self.start_pos - self.start_pos
            self.end_pos_norm = self.end_pos - self.start_pos

            self.movement_time = self.time[self.movement_ind]
            self.x_movement = self.x[self.movement_ind]
            self.y_movement = self.y[self.movement_ind]
            self.z_movement = self.z[self.movement_ind]
            self.x_movement_fit, self.movement_time_fit = self.b_spline_fit_1d(
                self.movement_time, self.x_movement, self.n_frames_fit, full_output=True)
            self.y_movement_fit = self.b_spline_fit_1d(self.movement_time, self.y_movement, self.n_frames_fit)
            self.z_movement_fit = self.b_spline_fit_1d(self.movement_time, self.z_movement, self.n_frames_fit)

            self.x_vel_movement = self.x_vel[self.movement_ind]
            self.y_vel_movement = self.y_vel[self.movement_ind]
            self.z_vel_movement = self.z_vel[self.movement_ind]
            self.x_vel_movement_fit = self.b_spline_fit_1d(self.movement_time, self.x_vel_movement, self.n_frames_fit)
            self.y_vel_movement_fit = self.b_spline_fit_1d(self.movement_time, self.y_vel_movement, self.n_frames_fit)
            self.z_vel_movement_fit = self.b_spline_fit_1d(self.movement_time, self.z_vel_movement, self.n_frames_fit)
        else:
            # in case the trajectory does not satisfy the movement initiation/termination criteria, in cases such as
            # when the participant number moved during the data collection period
            self.rt = np.nan
            self.mt = np.nan

            self.start_pos, self.end_pos = np.empty(3) * np.nan, np.empty(3) * np.nan,
            self.start_pos_norm, self.end_pos_norm = self.start_pos, self.end_pos

            self.movement_time = np.nan
            self.x_fit_trimmed = np.nan
            self.y_fit_trimmed = np.nan
            self.z_fit_trimmed = np.nan

            self.x_movement_fit, self.movement_time_fit = np.nan, np.nan
            self.y_movement_fit = np.nan
            self.z_movement_fit = np.nan

    def find_time(self, vel_threshold, ax=''):
        """
        :param vel_threshold: the velocity threshold, need to be in the same unit as the velocity
        :param ax: the axis used to determine movement initiation and termination. Default is the principal directions
        specified in the constructor. Other values are 'x', 'y', 'z', or 'xyz'
        :return:

        Find the movement initiation and termination time based on the velocity threshold and the velocity axis.
        """
        # determine the velocity axis to use.
        vel_all = np.concatenate([
            np.expand_dims(self.x_vel_smooth, 1),
            np.expand_dims(self.y_vel_smooth, 1),
            np.expand_dims(self.z_vel_smooth, 1),
        ], axis=1)

        if ax == '':  # default is the principal axes
            vel_principal = vel_all[:, self.principal_dir]

            # get the resultant velocity
            vel_eval = np.sqrt(vel_principal[:, 0] ** 2 + vel_principal[:, 1] ** 2)
        elif ax == 'x':
            vel_eval = np.abs(vel_all[:, 0])
        elif ax == 'y':
            vel_eval = np.abs(vel_all[:, 1])
        elif ax == 'z':
            vel_eval = np.abs(vel_all[:, 2])
        elif ax == 'xyz':
            vel_eval = np.sqrt(vel_all[:, 0] ** 2 + vel_all[:, 1] ** 2 + vel_all[:, 2] ** 2)
        else:
            raise ValueError('Unidentified axis! Enter either "" (an empty str), "x", "y", "z", or "xyz"')

        # find movement start and end times
        vel_threshold_ind = np.where(vel_eval >= vel_threshold)[0]

        if len(vel_threshold_ind) == 0:
            # in case there's no movement detected
            return np.nan, np.nan, np.nan

        vel_ind = self.consecutive(vel_threshold_ind)

        # in case there are multiple crossings at the threshold velocity
        if len(vel_ind) > 1:
            if self.movement_selection_method == 'length':
                # only use the portion of movement with the largest number of samples
                vel_len = [len(vel) for vel in vel_ind]
                max_vel = np.where(vel_len == np.max(vel_len))[0][0]
                vel_threshold_ind = vel_ind[max_vel]

            elif self.movement_selection_method == 'sign':
                displacement_all = np.concatenate([
                    np.expand_dims(self.x_smooth, 1),
                    np.expand_dims(self.y_smooth, 1),
                    np.expand_dims(self.z_smooth, 1),
                ], axis=1)

                displacement_eval = displacement_all[:, self.movement_selection_dir]

                movement_dist = []
                for vel in vel_ind:
                    movement_dist.append(displacement_eval[vel[-1]] - displacement_eval[vel[0]])
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

    def compute_transformation(self, screen_corners, full_output=False):
        """
        :param screen_corners: the corners of the surface on which the movement was performed
        :param full_output: whether to return full output, which includes the objects for the plane and corners
        :return:
        """
        # center the corners
        screen_center = np.mean(screen_corners, axis=0)
        screen_corners = screen_corners - screen_center

        # find the surface normal
        screen_plane = Plane.best_fit(screen_corners)
        screen_norm = Vector(screen_plane.cartesian()[:3])
        if screen_norm[self.ground_dir] < 0:
            screen_norm = screen_norm * -1

        # project the current screen normal to the ground plane
        screen_norm_ground = self.ground_plane.project_vector(screen_norm)

        # find the angle between the projected norm and the primary direction - this is to align the screen's primary
        # direction with the primary direction in the Cartesian coordinate
        primary_norm = np.zeros(3)
        primary_norm[self.primary_dir] = 1
        angle = screen_norm_ground.angle_between(primary_norm)
        angle = np.pi - angle if angle > np.pi / 2 else angle

        # construct the rotation matrix and rotate the screen normal
        ground_norm = np.zeros(3)
        ground_norm[self.ground_dir] = 1
        rotmat = matrix_from_axis_angle(np.hstack((ground_norm, (angle,))))
        rotation_to_align = Rotation.from_matrix(rotmat)
        screen_norm_rot = Vector(rotation_to_align.apply(screen_norm))

        # after the screen is aligned with the primary axis, we can rotate it around the secondary axis to make the
        # screen surface horizontal
        angle = -screen_norm_rot.angle_between(ground_norm)

        # axis is around the secondary direction
        secondary_norm = np.zeros(3)
        secondary_norm[self.secondary_dir] = 1
        rotmat = matrix_from_axis_angle(np.hstack((secondary_norm, (angle,))))
        rotation_to_ground = Rotation.from_matrix(rotmat)

        # the complete transformation - need to rotate to align first, then rotate to ground
        rotation = rotation_to_ground * rotation_to_align

        screen_corners_rot = Points(rotation.apply(screen_corners)) + screen_center
        screen_plane_rot = Plane.best_fit(screen_corners_rot)

        if full_output:
            return rotation, screen_center, {
                'screen_plane': screen_plane,
                'screen_corners_rot': screen_corners_rot,
                'screen_plane_rot': screen_plane_rot,
            }
        else:
            return rotation, screen_center

    def transform_data(self, x, y, z, screen_corners):
        """
        Spatially transform the trajectory so that the trajectory is on a flat, horizontal plane.
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        :param screen_corners: the corners of the screen
        :return:
        """
        rotation, screen_center = self.compute_transformation(screen_corners)
        coord = np.concatenate([np.expand_dims(x, axis=1),
                                np.expand_dims(y, axis=1),
                                np.expand_dims(z, axis=1)], axis=1)
        coord -= screen_center
        coord_rot = rotation.apply(coord)
        return coord_rot[:, 0], coord_rot[:, 1], coord_rot[:, 2]

    def find_start_and_end_pos(self, time_cutoff):
        """

        :param time_cutoff: The amount of time before/after the movement initiation/termination to consider when
        computing the trajectory's end positions. Instead of using the position at a particular time, the end position
        is the average of all positions within this time cutoff range.
        :return: mean_start, mean_end: the start and end positions
        """

        ind_start = (self.time < self.start_time) & (self.time > self.start_time - time_cutoff)
        if np.any(ind_start):
            start_x = self.x[ind_start]
            start_y = self.y[ind_start]
            start_z = self.z[ind_start]
            mean_start = np.mean(np.array([start_x, start_y, start_z]), axis=1)
        else:
            mean_start = np.empty((3,)) * np.nan

        ind_end = (self.time > self.end_time) & (self.time < self.end_time + time_cutoff)
        if np.any(ind_end) > 0:
            end_x = self.x[ind_end]
            end_y = self.y[ind_end]
            end_z = self.z[ind_end]
            mean_end = np.mean(np.array([end_x, end_y, end_z]), axis=1)
        else:
            mean_end = np.empty((3,)) * np.nan

        return mean_start, mean_end

    def validate_size(self):
        """ Validate input coordinate size. """
        n_x, n_y, n_z = len(self.x), len(self.y), len(self.z)
        if not (n_x == n_y == n_z):
            raise ValueError("The input x, y, and z have to be of the same size! \n"
                             f"Instead, len(x)={len(self.x)}, len(y)={len(self.y)}, len(z)={len(self.z)}")
        return n_x

    @staticmethod
    def consecutive(data, step_size=1):
        """ splits the missing data indices into chunks"""
        return np.split(data, np.where(np.diff(data) != step_size)[0] + 1)

    def missing_data(self):
        """
        Find if there is any outliers in the original movement trajectory. The outliers are detected by comparing each
        consecutive points' difference with a certain proportion of the overall range of motion.

        threshold: the threshold for the difference of x, y, and z, before and after the missing data block. If the
        difference is small, then we will use linear interpolation to fill in the gap. By default, the threshold for
        each axis is 1 mm/s.
        """
        # contain_anomolies = False
        missing_ind = np.where(self.x == self.missing_data_filler)[0]
        not_missing_ind = np.where(self.x != self.missing_data_filler)[0]
        if len(missing_ind) > 0:

            f_interp = interpolate.interp1d(self.time[not_missing_ind], self.x[not_missing_ind], bounds_error=False,
                                            fill_value=(np.NaN, np.NaN))
            self.x = f_interp(self.time)

            f_interp = interpolate.interp1d(self.time[not_missing_ind], self.y[not_missing_ind], bounds_error=False,
                                            fill_value=(np.NaN, np.NaN))
            self.y = f_interp(self.time)

            f_interp = interpolate.interp1d(self.time[not_missing_ind], self.z[not_missing_ind], bounds_error=False,
                                            fill_value=(np.NaN, np.NaN))
            self.z = f_interp(self.time)

            ind_delete = np.where(np.isnan(self.x))[0]
            self.x = np.delete(self.x, ind_delete)
            self.y = np.delete(self.y, ind_delete)
            self.z = np.delete(self.z, ind_delete)
            self.time = np.delete(self.time, ind_delete)
            self.n_frames = len(self.x)

            return True, len(missing_ind), missing_ind
        else:
            return False, 0, []

    def low_butter_scipy(self, signal, order=2):
        """
        Direct usage of the low-pass Butterworth Filter using library from SciPy.
        :param signal: 1D data to be filtered
        :param order: butterworth order
        :return: filtered signal
        """
        Wn = self.fc / (self.fs / 2)
        poly = butter(order, Wn, btype='lowpass', output='ba')  # returns numerator [0] and denominator [1] polynomials
        return filtfilt(poly[0], poly[1], signal.copy())

    @staticmethod
    def cent_diff(time, signal):
        """ Central difference method to find derivatives. """
        n_frames = len(time)
        der = np.zeros(n_frames, dtype=float)

        der[0] = (signal[1] - signal[0]) / (time[1] - time[0])
        der[-1] = (signal[-1] - signal[-2]) / (time[-1] - time[-2])

        for i_frame in np.arange(1, n_frames - 1):
            der[i_frame] = (signal[i_frame + 1] - signal[i_frame - 1]) / (
                    time[i_frame + 1] - time[i_frame - 1])

        return der

    @staticmethod
    def b_spline_fit_1d(time_vec, coord, n_fit, smooth=0., full_output=False):
        tck = interpolate.splrep(time_vec, coord,
                                 s=smooth,  # smoothing factor
                                 k=3,  # degree of the spline fit. Scipy recommends using cubic splines.
                                 )
        time_fit = np.linspace(np.min(time_vec), np.max(time_vec), n_fit)
        spline = interpolate.BSpline(tck[0], tck[1], tck[2])
        if full_output:
            return spline(time_fit), time_fit
        else:
            return spline(time_fit)

    def debug_plots(self, fig=None, axs=None):
        """ Create a debug plot that shows displacement, velocity, acceleration, and XY trajectory"""
        if axs is None:
            fig, axs = plt.subplots(2, 1)
            # plt.tight_layout()

        axs[0].plot(self.time_original, self.x_original, label='x', linestyle=':')
        axs[0].plot(self.time_original, self.y_original, label='y', linestyle=':')
        axs[0].plot(self.time_original, self.z_original, label='z', linestyle=':')
        axs[0].scatter(self.time_original[self.ind_missing], self.x_original[self.ind_missing], color='k')

        axs[0].plot(self.time, self.x_smooth, label='x', c='r')
        axs[0].plot(self.time, self.y_smooth, label='y', c='g')
        axs[0].plot(self.time, self.z_smooth, label='z', c='b')

        for principal_dir in self.principal_dir:
            axs[0].plot(self.time, np.ones((len(self.time), 1)) * self.start_pos[principal_dir],
                        c='c', linestyle=':', linewidth=3)
            axs[0].plot(self.time, np.ones((len(self.time), 1)) * self.end_pos[principal_dir],
                        c='m', linestyle=':', linewidth=3)

        axs[0].plot([self.start_time, self.start_time],
                    [np.min([self.x_smooth, self.y_smooth, self.z_smooth]),
                     np.max([self.x_smooth, self.y_smooth, self.z_smooth])], label='start', c='c')
        axs[0].plot([self.end_time, self.end_time],
                    [np.min([self.x_smooth, self.y_smooth, self.z_smooth]),
                     np.max([self.x_smooth, self.y_smooth, self.z_smooth])], label='end', c='m')
        axs[0].set_xlabel('Time (seconds)')
        axs[0].set_ylabel('Displacement')
        # axs[0].set_title(f'{self.n_missing} missing data')
        # axs[0].legend()

        axs[1].plot(self.time, self.x_vel_smooth, label='x', c='r')
        axs[1].plot(self.time, self.y_vel_smooth, label='y', c='g')
        axs[1].plot(self.time, self.z_vel_smooth, label='z', c='b')
        axs[1].plot([self.start_time, self.start_time],
                    [np.min([self.x_vel_smooth, self.y_vel_smooth, self.z_vel_smooth]),
                     np.max([self.x_vel_smooth, self.y_vel_smooth, self.z_vel_smooth])], label='start', c='c')
        axs[1].plot([self.end_time, self.end_time],
                    [np.min([self.x_vel_smooth, self.y_vel_smooth, self.z_vel_smooth]),
                     np.max([self.x_vel_smooth, self.y_vel_smooth, self.z_vel_smooth])], label='end', c='m')

        axs[1].set_xlabel('Time (seconds)')
        axs[1].set_ylabel('Velocity')
        # axs[1].set_title('Velocity')
        # axs[1].legend()

        plt.subplots_adjust(top=0.90, hspace=0.38, left=0.12, bottom=0.12)
        return fig, axs

    def demo_plots(self, fig=None, axs=None):
        if axs is None:
            fig, axs = plt.subplots(1, 1)
            # plt.tight_layout()

        axs.plot(self.time, self.x_smooth, label='x', linewidth=3, c='c')
        axs.plot(self.time, self.y_smooth, label='y', linewidth=3, c='m')
        axs.plot(self.time, self.z_smooth, label='z', linewidth=3, c='y')

        # axs.plot(self.time_original, self.x_original, label='x', linewidth=3, linestyle=':', c='k')
        # axs.plot(self.time_original, self.y_original, label='y', linewidth=3, linestyle=':', c='k')
        # axs.plot(self.time_original, self.z_original, label='z', linewidth=3, linestyle=':', c='k')
        # axs.scatter(self.time_original[self.ind_missing], self.x_original[self.ind_missing], color='k')

        axs.plot([self.start_time, self.start_time],
                 [np.min([self.x_smooth, self.y_smooth, self.z_smooth]),
                  np.max([self.x_smooth, self.y_smooth, self.z_smooth])],
                 label='Movement Start', color='g', linewidth=4, linestyle=':')
        axs.plot([self.end_time, self.end_time],
                 [np.min([self.x_smooth, self.y_smooth, self.z_smooth]),
                  np.max([self.x_smooth, self.y_smooth, self.z_smooth])],
                 label='Movement End', color='r', linewidth=4, linestyle=':')
        axs.set_xlabel('Time (s)', fontdict={'fontsize': 16})
        axs.set_ylabel('Displacement (mm)', fontdict={'fontsize': 16})
        axs.set_xticklabels(axs.get_xticklabels(), fontsize=12)
        axs.set_yticklabels(axs.get_yticklabels(), fontsize=12)
        # axs[0].set_title(f'Displacement, {self.n_missing} missing data')
        axs.legend()

        return fig, axs

    def debug_traj_3d(self, ax=None, color='k'):
        """ Create a debug plot that shows 3d trajectories"""
        if ax is None:
            plt.figure()
            ax = plt.axes(projection='3d')
        else:
            if ax.name != '3d':
                raise ValueError('The input MPL ax has to be a 3D axis! You can simply create an axis with the '
                                 'projection keyword: \n'
                                 'ax = plt.axes(projection=\'3d\')')

        # # ax.plot3D(self.x_smooth, self.y_smooth, self.z_smooth)
        # ax.scatter(self.x_smooth, self.y_smooth, self.z_smooth, c='k')
        # ax.scatter(self.x_smooth[0], self.y_smooth[0], self.z_smooth[0], c='r')  # beginning
        # ax.scatter(self.x_smooth[-1], self.y_smooth[-1], self.z_smooth[-1], c='g')  # end

        ax.plot3D(self.x, self.y, self.z, c=color)
        ax.scatter(self.x[0], self.y[0], self.z[0], c='r')  # beginning
        ax.scatter(self.x[-1], self.y[-1], self.z[-1], c='g')  # end

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        return ax

    def debug_traj_2d(self, x_axis, y_axis, var='', ax=None, center=True, full_output=False):
        """
        :param x_axis: the trajectory axis to be plotted on the x-axis of the 2d plot
        :param y_axis: the trajectory axis to be plotted on the y-axis of the 2d plot
        :param var: the variable to be plotted
        :param ax: matplotlib 2d axis object
        :param center: whether to center the data
        :param full_output: if True, will also return plot x and y coordinates
        :return: matplotlib 2d axis object
        """

        def check_axis(axis_name, _var):
            axis_name = axis_name.lower()

            if _var != '':
                _var = f'_{_var}'

            if (axis_name == 'x') | (axis_name == 'y') | (axis_name == 'z'):
                pos = getattr(self, f'{axis_name}{_var}')
            elif axis_name == 'xy':
                pos = np.sqrt((getattr(self, f'x{_var}') ** 2 +
                               getattr(self, f'y{_var}') ** 2))
            elif axis_name == 'xz':
                pos = np.sqrt((getattr(self, f'x{_var}') ** 2 +
                               getattr(self, f'z{_var}') ** 2))
            elif axis_name == 'yz':
                pos = np.sqrt((getattr(self, f'y{_var}') ** 2 +
                               getattr(self, f'z{_var}') ** 2))
            else:
                raise ValueError("Unrecognized trajectory axis name! Only support 1D or 2D axes!")
            return pos - pos[0] if center else pos

        plot_x = check_axis(x_axis, var)
        plot_y = check_axis(y_axis, var)
        line = ax.plot(plot_x, plot_y, alpha=.5)
        ax.scatter(plot_x[0], plot_y[0], c='r')
        ax.scatter(plot_x[-1], plot_y[-1], c='g')

        if full_output:
            return ax, line, plot_x, plot_y
        else:
            return ax, line

    @property
    def principal_dir(self):
        return self._principal_dir

    @principal_dir.setter
    def principal_dir(self, value):
        if value == 'xy':
            self._principal_dir = [0, 1]
        elif value == 'xz':
            self._principal_dir = [0, 2]
        elif value == 'yz':
            self._principal_dir = [1, 2]
        else:
            raise ValueError('Invalid principal directions! Please use the following: '
                             'xy, xz, or yz')

    @property
    def primary_dir(self):
        return self._primary_dir

    @primary_dir.setter
    def primary_dir(self, value):
        if value == 'x':
            self._primary_dir = 0
        elif value == 'y':
            self._primary_dir = 1
        elif value == 'z':
            self._primary_dir = 2
        else:
            raise ValueError('Invalid primary directions! Please use the following: '
                             'x, y, or z')

        self.secondary_dir = list(set(self.principal_dir) - {self.primary_dir})[0]

    @property
    def ground_dir(self):
        return self._ground_dir

    @ground_dir.setter
    def ground_dir(self, value):
        if value == 'x':
            self._ground_dir = 0
        elif value == 'y':
            self._ground_dir = 1
        elif value == 'z':
            self._ground_dir = 2
        else:
            raise ValueError('Invalid principal directions! Please use the following: '
                             'x, y, or z')

        ground_points = np.random.random((3, 3))
        ground_points[:, self._ground_dir] = 0
        self.ground_plane = Plane.best_fit(ground_points)
