"""
Sample data analysis script for the spatial cueing experiment in Wang & Welsh (2023). Written by X.M. Wang.

Wang, X.M., & Welsh, T.N. (2023). TAT-HUM: Trajectory Analysis Toolkit for Human Movements in Python.

"""

# %% import all the necessary packages

import os

# libraries for data manipulation
import pandas as pd
import numpy as np

# libraries for data visualization
import matplotlib.pyplot as plt

# libraries from TAT-HUM to process trajectory data
from tathum.functions import b_spline_fit_1d
from tathum.trajectory import Trajectory
from tathum.trajectory_mean import TrajectoryMean

# experiment-specific functions to process screen calibration
from demo.process_screen_calibration import process_screen_calibration

# %% analysis meta parameter setup

# specify the data directories
# NOTE: need to replace with the directories in which you store the data
param_path = '~/Downloads/data/trial_data'  # parameter files generated from the experiment
trajectory_path = '~/Downloads/data/trajectory_data'  # individual trajectory files for each trial

# booleans to determine whether to visualize 3D calibration results. Setting up a boolean controller at the beginning
# of the analysis script allows one to easily toggle different functionalities of the analysis using comment/uncomment
# plot_3d_calibration_check = True
plot_3d_calibration_check = False

# max number of missing trials before going to visual inspection
n_missing_max = 15

# experiment's principal movement directions - For different experiments, the principal axes might differ depending on
# the coordinate system established for the motion capture device
principal_ax = 'xz'

# target location in a 2d space - determined based on the experimental setup
target_location_2d = (
    [-140., 0., -np.sqrt(220. ** 2 - 140. ** 2)],
    [140., 0., -np.sqrt(220. ** 2 - 140. ** 2)],)

# a list of participants id's
par_id_all = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

# %% process screen calibration data using custom function

calibration_data_all, calibration_center_all = process_screen_calibration(
    trajectory_path, par_id_all, debug_plot=False, return_center=True)

# %% process trajectory data

# save name for the processed data
output_save_name = './demo/rt_mt_results.csv'

# a reference for processed trajectories - Sometimes, one may have already visually inspected the trials and determined which
# to keep and which to discard based on certain selection criteria. Afterwards, however, one may decide to examine
# additional dependent measures that requires them to go through all the trials again. In this scenario, one would only
# need to go through the trials that passed the visual inspection, instead of having to visually inspect all the trials
# again. Therefore, it would be useful to just use the processed data to automatically skip the trials that were
# discarded.
if os.path.exists(output_save_name):  # only read the reference file if it exists
    keep_trial_ref = pd.read_csv(output_save_name)
    processed_par = keep_trial_ref['par_id'].unique()
else:  # otherwise create placeholders
    keep_trial_ref = pd.DataFrame()
    processed_par = []

# variables to store the processed trial data (RT, MT, etc.)
output_df = pd.DataFrame()
output_df_ind = 0
# variables to store the processed trajectory data
trajectory_mean_all = []
trajectory_mean_reference = pd.DataFrame()
trajectory_ind = 0

# a figure for visual inspection for missing data
fig, axs = plt.subplots(2, 1)
plt.tight_layout()

# a figure for visual inspection for all trajectories
fig_traj, ax_traj = plt.subplots(1, 1)

# iterate all the participants
for par_id_ind, par_id in enumerate(par_id_all):

    # only create a figure if the user wants to visually inspect the spatially transformed trajectory data to make sure
    # the transformation was done correctly
    if plot_3d_calibration_check:
        fig_3d = plt.figure()
        ax_3d = plt.axes(projection='3d')

    # extract relevant screen calibration data obtained from the custom calibration processing function
    _, end_points = calibration_data_all[par_id_ind]
    screen_center = calibration_center_all[par_id_ind]

    # read the experimental parameter files for this participant
    param_name = f'{param_path}/par{par_id}_tathum.csv'
    param = pd.read_csv(param_name)

    # PsychoPy records all trials. We just want trials from the "trials" block
    param = param[~np.isnan(param['trials.thisN'])]

    # remove the first block
    param = param[param['trials.thisRepN'] > 0]

    # obtain the unique independent variable values to group trajectories later
    soa_all = np.sort(param['soa'].unique())
    target_all = np.sort(param['target'].unique())
    cue_all = np.sort(param['cue_ind'].unique())

    missing_data_count = 0

    par_processed = True if par_id in processed_par else False

    # loop through all the unique levels of different independent variables
    for soa_ind, soa in enumerate(soa_all):
        for target_ind, target in enumerate(target_all):
            target_location = target_location_2d[target_ind]
            for cue in cue_all:
                temp_param = param[(param['soa'] == soa) &
                                   (param['target'] == target) &
                                   (param['cue_ind'] == cue)]

                # determine the cue's congruency
                congruency = 'congruent' if cue == target else 'incongruent'

                # initialize a TrajectoryMean object to store movement trajectories for the current condition
                condition = {
                    'soa': soa,
                    'cue_location': cue,
                    'target_location': target,
                    'congruency': congruency}
                trajectory_mean = TrajectoryMean(exp_condition=condition)

                # create a dataframe to store all the trials with the same condition
                temp_output_df = pd.DataFrame()
                # this will be used to keep track of the trial index
                temp_output_df_ind = []
                for _, trial in temp_param.iterrows():

                    # skip trials that marked invalid during the experiment
                    if not trial['keep_trial']:
                        continue

                    trial_id = int(trial['trials.thisN'])

                    # see if this trial has been processed
                    if len(keep_trial_ref) > 0:
                        keep_trial_row = keep_trial_ref[(keep_trial_ref['par_id'] == par_id) &
                                                        (keep_trial_ref['trial_ind'] == trial_id)]
                    else:
                        keep_trial_row = {}  # use dict here so that later the IDE won't complain...

                    # a temporary output row to store all relevant data
                    temp_row = {
                        'par_id': par_id,
                        'trial_ind': trial_id,
                        'keep_trial': True,
                        'keep_trajectory': True,
                        'keep_both': True,
                        'soa': soa,
                        'cue_location': cue,
                        'target_location': target,
                        'congruency': congruency,
                        'mt': np.nan,
                        'rt': np.nan,
                        'end_pos_norm_x': np.nan,
                        'end_pos_norm_y': np.nan,
                        'end_pos_norm_z': np.nan,
                        'end_dist_to_target_2d': np.nan, }

                    trial_processed = False
                    if len(keep_trial_row) == 1:
                        trial_processed = True
                        keep_trial = keep_trial_row['keep_both'].values[0]

                        if not keep_trial:
                            temp_row['keep_trial'] = keep_trial_row['keep_trial']
                            temp_row['keep_trajectory'] = keep_trial_row['keep_trajectory']
                            temp_row['keep_both'] = keep_trial_row['keep_both']
                            temp_output_df = pd.concat([temp_output_df, pd.DataFrame(temp_row, index=[0])],
                                                       ignore_index=True)
                            continue  # skip this trial if previously decided so

                    # trajectory file's path
                    file_name = f'{trajectory_path}/par{par_id}/par{par_id}_trial_{trial_id}.csv'

                    try:
                        # read the trajectory data
                        raw_data = pd.read_csv(file_name)
                    except:
                        # this could happen when the data file is empty due to software glitch
                        print(f'file {file_name} does not contain valid data!')
                        missing_data_count += 1
                        temp_row['keep_trial'] = False
                        temp_output_df = pd.concat([temp_output_df, pd.DataFrame(temp_row, index=[0])],
                                                   ignore_index=True)
                        continue

                    # assign column names for easy reference
                    raw_data.columns = ('time', 'frame', 'x', 'y', 'z')

                    # initialize a Trajectory object using this trial's trajectory data
                    trajectory = Trajectory(raw_data.x, raw_data.y, raw_data.z,
                                            principal_dir=principal_ax,
                                            time=raw_data.time, fs=250, fc=10,
                                            transform_end_points=end_points)

                    keep_trial = True
                    # determine whether to keep this trial based on the number of missing data due to marker occlusion
                    if trajectory.contain_missing & (trajectory.n_missing > n_missing_max) & (not trial_processed):
                        missing_data_count += 1

                        # create a debug plot to visually inspect the missing data. If the most of the missing data
                        # occurred outside the movement segment, one can opt to keep the trial.
                        trajectory.debug_plots(fig=fig, axs=axs)
                        plt.suptitle(f'participant {par_id}, trial {trial_id}, n missing data = {trajectory.n_missing}')
                        plt.pause(0.1)
                        response = input('keep entry. Enter if yes, "n" if no. ')
                        axs[0].cla()
                        axs[1].cla()
                        if response == 'n':
                            keep_trial = False

                    # automatically discard the trial if the trial does not contain any movement (e.g., the participant
                    # failed to move within the data collection period).
                    if not trajectory.contain_movement:
                        temp_row['keep_trial'] = False

                    # store all the relevant values if we decide to keep the trial
                    if keep_trial:
                        trajectory_mean.add_trajectory(trajectory)
                        temp_row['mt'] = trajectory.mt
                        temp_row['rt'] = trajectory.rt
                        temp_row['end_pos_norm_x'] = trajectory.end_pos_norm[0]
                        temp_row['end_pos_norm_y'] = trajectory.end_pos_norm[1]
                        temp_row['end_pos_norm_z'] = trajectory.end_pos_norm[2]
                        # this is the Euclidean distance between the end point and the target
                        temp_row['end_dist_to_target_2d'] = np.sqrt(
                            (trajectory.end_pos_norm[0] - target_location[0]) ** 2 +
                            (trajectory.end_pos_norm[2] - target_location[2]) ** 2)

                    # plot the 3D view of the original trajectories and the spatially transformed trajectories to make
                    # sure the spatial transformation is done properly - it's always good to visually check the
                    # operations
                    if plot_3d_calibration_check:
                        ax_3d.plot(raw_data.x, raw_data.y, raw_data.z)
                        ax_3d.plot(trajectory.x_movement_fit, trajectory.y_movement_fit, trajectory.z_movement_fit)
                        ax_3d.scatter(end_points[:, 0] + screen_center[0],
                                      end_points[:, 1] + screen_center[1],
                                      end_points[:, 2] + screen_center[2])
                        ax_3d.set_xlabel('x')
                        ax_3d.set_ylabel('y')
                        ax_3d.set_zlabel('z')
                        ax_3d.set_title(f'participant {par_id}')

                    # store the output of this trial with the appropriate trial index (so that later on when we look for
                    # trajectory outliers, we would know which trial to refer to).
                    temp_output_df = pd.concat(
                        [temp_output_df, pd.DataFrame(temp_row, index=[output_df_ind])], ignore_index=True)
                    temp_output_df_ind.append(output_df_ind)
                    output_df_ind += 1

                # use the Euclidean distance between the end point and the target to identify any potential outliers
                # where the participants did not reach the target
                end_dist_mean = np.mean(temp_output_df['end_dist_to_target_2d'])
                end_dist_sd = np.std(temp_output_df['end_dist_to_target_2d'])
                outliers_ind = np.where(
                    (temp_output_df['end_dist_to_target_2d'] > end_dist_mean + 3 * end_dist_sd) |
                    (temp_output_df['end_dist_to_target_2d'] < end_dist_mean - 3 * end_dist_sd))[0]
                if len(outliers_ind) > 0:
                    # remove the trial if it is an outlier
                    ind = temp_output_df.index[outliers_ind]
                    temp_output_df['keep_trial'][ind] = False
                    # also need to remember to remove its trajectory data before computing the mean trajectory
                    trajectory_mean.remove_trajectory(outliers_ind)

                # after removing the end point outliers, finally store the processed data to the final output df
                output_df = pd.concat([output_df, temp_output_df], ignore_index=True)

                # finally can compute the mean trajectory
                trajectory_mean.compute_mean_trajectory()

                # also visually compare the mean trajectory with the individual trajectories in case there is any
                # abnormal trials if the participant has not been processed previously
                if not par_processed:
                    # plot the trajectories
                    trajectory_mean.debug_plots_trajectory(fig=fig_traj, ax=ax_traj)

                    # use figure title to provide more information
                    plt.suptitle(f'Participant {par_id}, SOA = {soa}, Target = {target}, Cue = {cue}')
                    plt.pause(0.1)
                    response = input('type in the trajectories that you would like to exclude, separate by comma ')
                    ax_traj.cla()

                    if len(response) > 0:
                        # remove the trial
                        outliers_ind = [int(ind) for ind in response.split(', ') if ind != '']
                        trajectory_mean.remove_trajectory(outliers_ind)

                        # double check
                        trajectory_mean.debug_plots_trajectory(fig=fig_traj, ax=ax_traj)
                        plt.pause(0.1)
                        response = input('Double check. Press enter to continue')
                        ax_traj.cla()

                        for outlier in outliers_ind:
                            output_df['keep_trajectory'][temp_output_df_ind[outlier]] = False

                        # print(output_df['keep_trajectory'][temp_output_df_ind])
                        print(f'for participant {par_id}, {len(outliers_ind)} trajectories were removed!')

                # finally, store the mean trajectory
                trajectory_mean_all.append(trajectory_mean)
                temp_row = {
                    'par_id': par_id,
                    'soa': soa,
                    'cue_location': cue,
                    'target_location': target,
                    'congruency': congruency,
                    'trajectory_ind': trajectory_ind, }
                trajectory_mean_reference = pd.concat([trajectory_mean_reference, pd.DataFrame(temp_row, index=[0])],
                                                      ignore_index=True)
                trajectory_ind += 1

    print(f'Participant {par_id}: Discarded {missing_data_count} trials due to missing data!')

# convert the time from seconds to miliseconds
output_df['rt'] = output_df['rt'] * 1000
output_df['mt'] = output_df['mt'] * 1000

output_df['keep_both'] = output_df['keep_trial'] & output_df['keep_trajectory']
output_df.to_csv(output_save_name)

# %% analyze trajectory congruency area

trajectory_out_save_name = './demo/mean_area_results.csv'

# first get all the unique values
par_id_all = trajectory_mean_reference['par_id'].unique()
soa_all = trajectory_mean_reference['soa'].unique()
cue_location_all = trajectory_mean_reference['cue_location'].unique()
target_location_all = trajectory_mean_reference['target_location'].unique()
congruency_all = trajectory_mean_reference['congruency'].unique()

# toggle the bool to visualize trajectories
# plot_trajectory = True
plot_trajectory = False

# the amount of normalized distance to consider when calculating the congruency area
dist_perc = 50

# the portions of the trajectory whose area we will calculate
sub_trajectory_ind = [[0, 19], [20, 39], [40, 59], [60, 79], [80, 99]]

# number of fits for the b-spline
n_spline_fit = 100

# empty data frames to store relevant output
mean_trajectory_summary = pd.DataFrame()
mean_area = pd.DataFrame()
for par_id in par_id_all:

    if plot_trajectory:
        # create a new figure for each participant only when we need to plot the trajectories
        fig, ax = plt.subplots(3, 2)
        plt.suptitle(f'participant {par_id}')

    # loop through all the trials
    for i_target, target_location in enumerate(target_location_all):
        for i_soa, soa in enumerate(soa_all):
            temp_reference = trajectory_mean_reference[
                (trajectory_mean_reference['par_id'] == par_id) &
                (trajectory_mean_reference['target_location'] == target_location) &
                (trajectory_mean_reference['soa'] == soa)]

            # create a temporary matrix to store all the trajectories from the same condition (easy averaging)
            temp_x_comp = np.zeros([len(trajectory_mean_all[0].x_mean), 2])
            # this will be used to store the b-spline objects for the congruent and incongruent trials so that we can
            # perform integration to obtain their respective areas
            temp_spline_comp = np.empty(2, dtype=object)

            # loop through all the trials for the specific condition
            for i_row, row in temp_reference.iterrows():
                congruency = row.congruency
                cue_location = row.cue_location
                traj_ind = row.trajectory_ind
                trajectory_mean = trajectory_mean_all[traj_ind]

                # the z-direction is inverted for participants 7 and 8 and therefore, need to invert them
                invert_z = -1 if (par_id == 7) | (par_id == 8) else 1

                # to derive the congruency area (trajectory area between the congruent and incongruent trials), we need
                # to ensure that we are always performing the subtraction in the same order. In this case, we can just
                # always use congruent - incongruent
                col = 1 if congruency == 'congruent' else 0

                # invert x axis if target == 1
                trajectory_sign = -1 if target_location == 1 else 1
                trajectory_mean.x_mean *= trajectory_sign

                # now store the mean trajectory
                temp_x_comp[:, col] = trajectory_mean.x_mean

                # parametrize the mean trajectory. Because the TrajectoryMean class does not have a built-in B-spline
                # method, we just need to use the version from the Function module.
                # NOTE: because the trajectory mean does not have a corresponding time vector, we will just use a
                # vector of continuous integers instead
                _, _, temp_spline_comp[col] = b_spline_fit_1d(
                    np.arange(len(trajectory_mean.x_mean)), trajectory_mean.x_mean, n_spline_fit, return_spline=True)

                # store all the parameterized trajectory points in one single data frame
                n_points = len(trajectory_mean.x_mean)
                temp_summary = pd.DataFrame({
                    'par_id': np.ones((n_points)) * par_id,
                    'target': np.ones((n_points)) * target_location,
                    'soa': np.ones((n_points)) * soa,
                    'congruency': np.tile(congruency, (n_points)),
                    'time': np.arange(n_points),
                    'x_mean': trajectory_mean.x_mean * trajectory_sign,
                    'x_sd': trajectory_mean.x_sd,
                    'z_mean': trajectory_mean.z_mean,
                    'z_sd': trajectory_mean.z_sd,
                }, index=np.arange(n_points))
                mean_trajectory_summary = pd.concat([mean_trajectory_summary, temp_summary], ignore_index=True)

                if plot_trajectory:
                    c = 'g' if congruency == 'congruent' else 'r'

                    # one can selectively plot different aspects of the trajectory for visually inspect or just to
                    # understand the data a bit more
                    plot_time = 50  # can choose to only plot portion of the trajectory

                    # plot all the individual trajectories
                    for traj_single in trajectory_mean.all_trajectories:
                        ax[i_soa][i_target].plot(
                            traj_single.x_movement_fit[:plot_time] * trajectory_sign,
                            traj_single.z_movement_fit[:plot_time] * invert_z,
                            color=c, alpha=.2, )

                    # plot the mean trajectory
                    ax[i_soa][i_target].plot(
                        trajectory_mean.x_mean[:plot_time] * trajectory_sign,
                        trajectory_mean.z_mean[:plot_time] * invert_z,
                        color=c, alpha=.2, )
                    ax[i_soa][i_target].fill_between(
                        trajectory_mean.x_mean[:plot_time] * trajectory_sign,
                        trajectory_mean.z_mean[:plot_time] * invert_z - trajectory_mean.z_sd[:plot_time],
                        trajectory_mean.z_mean[:plot_time] * invert_z + trajectory_mean.z_sd[:plot_time],
                        color=c, alpha=.2, )

                    ax[i_soa][i_target].set_title(f'soa = {soa}, target = {target_location}')

            # compute the total area between the congruent and incongruent trials based on discrete data points
            area = np.sum(temp_x_comp[:, 1] - temp_x_comp[:, 0])

            # compute the continuous total area using the built-in integration function of the B-spline class
            area_continuous = temp_spline_comp[1].integrate(0, 99) - temp_spline_comp[0].integrate(0, 99)

            # extract different segments of the trajectory
            for ind in sub_trajectory_ind:
                area_sub_discrete = np.sum(temp_x_comp[ind[0]:ind[1], 1] - temp_x_comp[ind[0]:ind[1], 0])

                ind_start, ind_end = ind[0], ind[1]
                area_sub_continuous = temp_spline_comp[1].integrate(ind_start, ind_end) - temp_spline_comp[0].integrate(
                    ind_start, ind_end)

                temp_row = {
                    'par_id': par_id,
                    'soa': soa,
                    'target_location': target_location,
                    'congruency_area': area,
                    'congruency_area_continuous': area_continuous,
                    'congruency_area_absolute': np.abs(area),
                    'sub_ind': ind[1],
                    'sub_area': area_sub_discrete,
                    'sub_area_continuous': area_sub_continuous, }
                mean_area = pd.concat([mean_area, pd.DataFrame(temp_row, index=[0])], ignore_index=True)

mean_area.to_csv(trajectory_out_save_name)
