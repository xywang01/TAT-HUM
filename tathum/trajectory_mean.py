"""
The TrajectoryMean class which allows storing a collection of individual trajectories to derive the characteristics of
the mean trajectory.

Written by X.M. Wang.

Wang, X.M., & Welsh, T.N. (2023). TAT-HUM: Trajectory Analysis Toolkit for Human Movements in Python.
"""
from typing import Optional
import matplotlib.pyplot as plt
from .trajectory_base import TrajectoryBase
from .coord import Coord
import numpy as np


class TrajectoryMean:
    x = Coord()
    y = Coord()
    z = Coord()
    time = Coord()

    def __init__(self, exp_condition: Optional[dict] = None):
        self.all_trajectories = []
        self.exp_condition = exp_condition
        self.x_mean = None
        self.y_mean = None
        self.z_mean = None
        self.x_sd = None
        self.y_sd = None
        self.z_sd = None

    @property
    def n_trajectories(self):
        return len(self.all_trajectories)

    def add_trajectory(self, traj: TrajectoryBase):
        """
        Add a trajectory to the TrajectoryMean object.
        :param traj: The trajectory to add.
        :return: None
        """
        self.all_trajectories.append(traj)

    def compute_mean_trajectory(self,
                                traj_names=('x_fit', 'y_fit', 'z_fit'),
                                post_script=''):
        """
        Compute the mean trajectory for the current condition.
        :param traj_names: The names of the trajectory attributes to compute the mean trajectory.
        :param post_script: The post script to append to the mean trajectory attribute names, if needed.
        :return: None
        """
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
            sd = np.std(temp, axis=1)  # / np.sqrt(temp.shape[1])
            self.__setattr__(mean_name, mean)
            self.__setattr__(sd_name, sd)

    def remove_trajectory(self, ind_list):
        """
        Remove trajectories from the TrajectoryMean object.
        :param ind_list: The indices of the trajectories to remove.
        :return: None
        """
        for ind in sorted(ind_list, reverse=True):  # loop in reverse order to not throw off the subsequent indices
            self.all_trajectories.pop(ind)

    def debug_plots_trajectory(self,
                               principal_dir='xy',
                               single_name_generic='_fit',
                               mean_name_generic='_mean',
                               fig=None, ax=None,
                               show_text=False,):
        """
        Plot the individual trajectories and the mean trajectory.
        :param principal_dir: The principal directions to plot the trajectories.
        :param single_name_generic: The generic name of the trajectory attributes to plot the individual trajectories.
        :param mean_name_generic: The generic name of the trajectory attributes to plot the mean trajectory.
        :param fig: The figure to plot the trajectories.
        :param ax: The axes to plot the trajectories.
        :param show_text: Whether to show the text on the plot.
        :return: The figure and axes.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        for ind, trajectory in enumerate(self.all_trajectories):
            plt_x = trajectory.__getattribute__(f'{principal_dir[0]}{single_name_generic}')
            plt_y = trajectory.__getattribute__(f'{principal_dir[1]}{single_name_generic}')
            n_plot = len(plt_x)
            ax.plot(plt_x, plt_y, pickradius=5, alpha=.7)
            ax.text(plt_x[int(n_plot / 2)], plt_y[int(n_plot / 2)], str(ind))

        mean_x = self.__getattribute__(f'{principal_dir[0]}{mean_name_generic}')
        mean_y = self.__getattribute__(f'{principal_dir[1]}{mean_name_generic}')

        ax.plot(mean_x, mean_y, linewidth=8, alpha=.5, color='b', pickradius=2)

        ax.scatter(mean_x[0], mean_y[0], marker='o', color='g', s=100)
        ax.scatter(mean_x[-1], mean_y[-1], marker='o', color='r', s=100)

        if show_text:
            ax.text(mean_x[0], mean_y[0], 'START', color='g', fontsize=20)
            ax.text(mean_x[-1], mean_y[-1], 'END', color='r', fontsize=20)

        ax.set_xlabel('x displacement')
        ax.set_ylabel('y displacement')

        return fig, ax
