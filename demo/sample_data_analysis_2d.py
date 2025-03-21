"""
Sample data analysis for 2D movements. This example uses Optotrak data with movement performed on a 2D surface. An
alternative example is provided in demo/sample_data_analysis_kinarm_2d.py, which uses Kinarm data.

Written by X.M. Wang.

Wang, X.M., & Welsh, T.N. (2023). TAT-HUM: Trajectory Analysis Toolkit for Human Movements in Python.
"""

import numpy as np
import pandas as pd

from tathum.functions import Preprocesses
from tathum.trajectory_2d import Trajectory2D
from tathum.trajectory_mean import TrajectoryMean

# NOTE: The sample 2D data is currently unavailable and will be uploaded later.
# Load the data. The raw data is grouped into a single csv file. The transform data is also grouped into a single csv
# file.
raw_data = pd.read_csv(f'./demo/demo_data/data_2d/demo_data_2d.csv')
transform_data = pd.read_csv(f'./demo/demo_data/data_2d/transform_data_2d.csv')

# %%
# get all levels of different variables, based on which the data will be grouped
par_id_all = raw_data['par_id'].unique()
vision_all = raw_data['vision'].unique()

# parse the transform data for easy access
transform_lookup = {}
for par_id in par_id_all:
    # get the transform data for the current participant
    temp_transform = transform_data[(transform_data['par_id'] == par_id)]
    exp_end_point = temp_transform[['x', 'y']].to_numpy()
    exp_end_point = (exp_end_point[0], exp_end_point[1])  # reformat the data for the appropriate input type

    # assign the transform data to the lookup table as a dictionary object
    transform_lookup[par_id] = exp_end_point

# create dataframes to store output
dv_output = pd.DataFrame()
trajectory_mean_all = []
trajectory_mean_reference = pd.DataFrame()
trajectory_mean_idx = 0  # this is used to keep track of the index of the trajectory mean object based on conditions

# iterate through all conditions
for vision in vision_all:
    # Because there is no repetitions, group trials based on the participant id
    # initialize a TrajectoryMean object to store movement trajectories for the current condition
    condition = {'vision': vision}

    trajectory_mean = TrajectoryMean(exp_condition=condition)

    for par_id in par_id_all:
        # get the experimental end point to perform spatial transformation (rotation)
        exp_end_point = transform_lookup[par_id]

        # get the data for the current condition
        df_idx = np.where((raw_data['par_id'] == par_id) &
                          (raw_data['vision'] == vision))[0]
        temp = raw_data.iloc[df_idx]
        temp_x = temp['x'].values
        temp_y = temp['y'].values

        # create a Trajectory2D object to store the movement trajectory for the current trial
        traj = Trajectory2D(
            x=temp_x,
            y=temp_y,
            fs=60,  # Hz, time stamp is not available from the dataset

            transform_end_points=exp_end_point,
            transform_to=np.array([0, 1]),

            displacement_preprocess=(Preprocesses.LOW_BUTTER, ),
            velocity_preprocess=(Preprocesses.CENT_DIFF, Preprocesses.LOW_BUTTER, ),
            acceleration_preprocess=(Preprocesses.CENT_DIFF, Preprocesses.LOW_BUTTER, ),
        )

        # output and store all the dependent measures - This is achieved through a built-in function of the
        # Trajectory2D class
        temp_kin = traj.format_results()
        # add condition information to the output
        temp_kin['par_id'] = par_id
        temp_kin['vision'] = vision
        # add the output to the dataframe
        dv_output = pd.concat([dv_output, temp_kin], ignore_index=True)

        # add the trajectory to the TrajectoryMean object
        trajectory_mean.add_trajectory(traj)

    # compute the mean trajectory for the current condition
    # NOTE: The default compute mean trajectory takes all three axis. However, because we only use two axes, we
    # need to manually specify the axes to compute the mean trajectory.
    trajectory_mean.compute_mean_trajectory(traj_names=('x_fit', 'y_fit'),)
    trajectory_mean_all.append(trajectory_mean)

    # store the index of the trajectory mean object for the current condition for future reference
    temp_ref = pd.DataFrame({
        'vision': vision,
        'trajectory_mean_idx': trajectory_mean_idx
    }, index=[0])
    trajectory_mean_reference = pd.concat([trajectory_mean_reference, temp_ref], ignore_index=True)
    trajectory_mean_idx += 1

# %% plot the dependent measures
import matplotlib.pyplot as plt
import seaborn as sns

# list all relevant dependent measures
dv_all = (
    # 'rt', 'mt', 'movement_dist',  # because of how data collection is setup, RT, MT, and movement distance are not available
    'peak_vel', 'time_to_peak_vel', 'time_after_peak_vel',
    'peak_acc', 'time_to_peak_acc', 'time_after_peak_acc')

# iterate through all dependent measures and plot them
for dv in dv_all:
    sns.catplot(
        data=dv_output,

        x='vision',
        y=dv,

        kind='bar',
    )
    plt.suptitle(dv)

# %% plot the mean trajectory
# different colors correspond to different vision conditions - red for preview, green for full
traj_colors = ('r', 'g')

# create a figure with subplots for each condition
fig_traj, ax_traj = plt.subplots(1, 1)

for i_vis, vision in enumerate(vision_all):
    # get the mean trajectory for the current condition
    traj_mean_idx = trajectory_mean_reference[
        (trajectory_mean_reference['vision'] == vision)]['trajectory_mean_idx'].values[0]
    traj_mean = trajectory_mean_all[traj_mean_idx]

    # plot the mean trajectory
    ax_traj.plot(traj_mean.x_mean, traj_mean.y_mean, traj_colors[i_vis])

    # add shaded error bars along the x-axis
    ax_traj.fill_betweenx(
        traj_mean.y_mean,
        traj_mean.x_mean - traj_mean.x_sd,
        traj_mean.x_mean + traj_mean.x_sd,
        color=traj_colors[i_vis],
        alpha=0.2,  # Adjust alpha for transparency
    )

    # add figure objects
    ax_traj.set_title('Trajectory Comparison: Preview vs. Full Vision')
    ax_traj.set_xlabel('x')
    ax_traj.set_ylabel('y')

    # scale the axes limits
    ax_traj.axis('equal')
