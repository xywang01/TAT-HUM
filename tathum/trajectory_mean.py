"""
The TrajectoryMean class which allows storing a collection of individual trajectories to derive the characteristics of
the mean trajectory.

Written by X.M. Wang.

Wang, X.M., & Welsh, T.N. (2023). TAT-HUM: Trajectory Analysis Toolkit for Human Movements in Python.

"""

import matplotlib.pyplot as plt

from .trajectory import Trajectory
from .coord import Coord

import numpy as np


class TrajectoryMean:
    x = Coord()
    y = Coord()
    z = Coord()
    time = Coord()

    def __init__(self, exp_condition: dict):  # x, y, z, x_sd, y_sd, z_sd,
        self.all_trajectories = []
        self.exp_condition = exp_condition
        self.x_mean = None
        self.y_mean = None
        self.z_mean = None
        self.x_sd = None
        self.y_sd = None
        self.z_sd = None

    def add_trajectory(self, traj: Trajectory):
        self.all_trajectories.append(traj)

    def compute_mean_trajectory(self,
                                traj_names=('x_movement_fit', 'y_movement_fit', 'z_movement_fit'),
                                post_script=''):
        for name in traj_names:
            coord_name = name[0]
            mean_name = f'{coord_name}{post_script}_mean'
            sd_name = f'{coord_name}{post_script}_sd'

            temp = None
            for ind, traj in enumerate(self.all_trajectories):
                temp_coord = traj.__getattribute__(name)

                # center the coordinate's starting position
                temp_coord -= temp_coord[0]

                if len(temp_coord.shape) < 2:
                    temp_coord = np.expand_dims(temp_coord, 1)

                if temp is None:
                    temp = temp_coord.copy()
                else:
                    temp = np.append(temp, temp_coord, axis=1)

            mean = np.mean(temp, axis=1)
            sd = np.std(temp, axis=1)   # / np.sqrt(temp.shape[1])
            self.__setattr__(mean_name, mean)
            self.__setattr__(sd_name, sd)

    def remove_trajectory(self, ind_list):
        for ind in sorted(ind_list, reverse=True):  # loop in reverse order to not throw off the subsequent indices
            self.all_trajectories.pop(ind)

    def debug_plots_trajectory(self, principal_dir='xz',
                                      single_name_generic='_movement_fit',
                                      mean_name_generic='_mean',
                                      fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        print(f'There are {len(self.all_trajectories)} trajectories!')
        for ind, trajectory in enumerate(self.all_trajectories):
            plt_x = trajectory.__getattribute__(f'{principal_dir[0]}{single_name_generic}')
            plt_y = trajectory.__getattribute__(f'{principal_dir[1]}{single_name_generic}')
            n_plot = len(plt_x)
            ax.plot(plt_x, plt_y, pickradius=5)
            ax.text(plt_x[int(n_plot/2)], plt_y[int(n_plot/2)], str(ind))

        ax.plot(self.__getattribute__(f'{principal_dir[0]}{mean_name_generic}'),
                self.__getattribute__(f'{principal_dir[1]}{mean_name_generic}'),
                linewidth=20, alpha=.5, color='g', pickradius=5)

        ax.set_xlabel('x displacement')
        ax.set_ylabel('y displacement')

        return fig, ax
