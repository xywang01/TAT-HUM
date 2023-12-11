"""
This is a demo script to show how to use the Trajectory2D class to analyze 2D movement data from Kinarm.

Written by X.M. Wang.

Wang, X.M., & Welsh, T.N. (2023). TAT-HUM: Trajectory Analysis Toolkit for Human Movements in Python.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

from tathum.functions import Preprocesses, consecutive, rotation_mat_2d
from tathum.trajectory_2d import Trajectory2D
from tathum.trajectory_mean import TrajectoryMean

# IMPORTANT: need to import KinArm's custom Python reader, which can be downloaded from:
# https://kinarm.com/task/kinarm-exam-file-reading-in-python/
# After download, you can directly copy the folder (KinarmReaderPython) to your project folder and call the import
# statement below
import sys
sys.path.append('/Users/michael/GitHub/tathum-kinarm')  # optionally add the path to the KinarmReaderPython folder if not in the same directory
from KinarmReaderPython.exam_load import ExamLoad


# event codes for different movements markers based on Kinarm output
event_code = {
    'stim_on': 'E_TARGET2_ON',
    'movement_start': 'E_START_MT',
    'movement_end': 'E_TARGET2_REACHED',
}

# different trial types based on experimental design
trial_type_all = [13, 14, 15, 16]

# create a mean trajectory object for each trial type
trajectory_mean_all = [TrajectoryMean({'trial_type': trial_type}) for trial_type in trial_type_all]

# load data
data_path = './demo/demo_data/data_2d_kinarm'

# A custom task
file_name = 'kinarm_custom_task.kinarm'

# read the data using the KinarmReaderPython package
raw_data = ExamLoad(f'{data_path}/{file_name}')

# extract experimental parameters from the loaded Kinarm data
exp_summary = raw_data.trials['common'].parameters
target_rotation = np.array(exp_summary['TARGET_TABLE:Rotation'])
target_x = np.array(exp_summary['TARGET_TABLE:X']) / 100
target_x_global = np.array(exp_summary['TARGET_TABLE:X_GLOBAL']) / 100
target_y = np.array(exp_summary['TARGET_TABLE:Y']) / 100
target_y_global = np.array(exp_summary['TARGET_TABLE:Y_GLOBAL']) / 100

# index lookup for Trial Protocol (trial type (13, 14, 15, 16) references this table, which, in turn, selects the
# corresponding target location index). Because the indices are from MATLAB, need to subtract 1 to get the correct
tp_start_idx = np.array(exp_summary['TP_TABLE:Start Target']) - 1
tp_end_idx = np.array(exp_summary['TP_TABLE:End Target']) - 1

# get the home position for the current participant
home_pos = np.array([target_x_global[0], target_y_global[0]])

# create a figure to plot the trajectory during debug
fig_debug, ax_debug = plt.subplots(2, 1)

# iterate through all the trials based on KinArm Python reader's default data structure
trial_summary = pd.DataFrame()
for trial in raw_data.trials.values():
    # skipping empty trials
    if (trial.name == '') | (trial.ack_count == 0):
        continue

    # find the trial type and its corresponding mean trajectory index
    _, trial_type, trial_rep = trial.name.split('_')
    # get the corresponding index for the trial type to store the mean trajectory
    trial_type_idx = trial_type_all.index(int(trial_type))
    # because of the trial type indices are from MATLAB, need to convert to Python's index
    trial_type = int(trial_type) - 1

    # get the target start and end position indices
    target_start_idx = tp_start_idx[trial_type]
    target_end_idx = tp_end_idx[trial_type]

    # get the target start and end position in the global coordinate
    target_start = np.array([target_x_global[target_start_idx], target_y_global[target_start_idx]])
    target_end = np.array([target_x_global[target_end_idx], target_y_global[target_end_idx]])

    # get the relevant trial protocol to extract the trial parameters
    tp_num = trial.parameters['TRIAL:TP_NUM'][0]
    tp_row = trial.parameters['TRIAL:TP_ROW'][0]
    trial_num = trial.parameters['TRIAL:TRIAL_NUM'][0]
    run_count = trial.parameters['TRIAL:TRIAL_RUN_COUNT'][0]

    # store the trial summary
    temp_summary = pd.DataFrame({
        'trial_name': trial.name,
        'trial_type': trial_type,
        'repetition': trial_rep,
        'trial_num': trial_num,
        'run_count': run_count,
    }, index=[0])

    # extract the Kinarm events and their timings
    events_all = {e.label: e.time for e in trial.events}

    # convert the movement data into numpy array
    right_pos = np.array(trial.positions['Right_Hand'].values)
    # left_pos = np.array(trial.positions['Left_Hand'].values)  # not relevant in this example

    # rotate the movement data based on the target rotation angle
    rot_angle = target_rotation[target_end_idx]
    rotation_mat = rotation_mat_2d(np.deg2rad(-rot_angle))
    right_pos = np.matmul(right_pos - home_pos, rotation_mat) + home_pos

    # because Kinarm does not output the time stamp, need to create a time stamp based on the frame rate.
    # one can also alternatively just input the frame_rate when initializing the Trajectory2D class. However, I did not
    # do this here because I want to use the timestamps to pre-trim the movement data based on the stimulus onset time
    # from the event
    timestamps = np.arange(trial.frame_count) / trial.frame_rate

    # use Kinarm's event code to extract the movement onset and offset as the custom movement selection function.
    stim_on_time = events_all[event_code['stim_on']]
    movement_start_time = events_all[event_code['movement_start']]
    movement_end_time = events_all[event_code['movement_end']]

    # trim the movement data based on the stimulus onset time (for RT calculation in the Trajectory class)
    stim_on_ind = np.where(timestamps >= stim_on_time)[0][0]
    right_pos = right_pos[stim_on_ind:]
    # left_pos = left_pos[stim_on_ind:]  # not relevant in this example
    timestamps = timestamps[stim_on_ind:]

    # identify the movement onset and offset indices in the trimmed data
    movement_idx = np.where((timestamps >= movement_start_time) & (timestamps <= movement_end_time))[0]

    # create a custom function that outputs the movement onset and offset time and indices. This simply returns the
    # values that we identified above using Kinarm's event codes. This is a good example of using extra variables to
    # customize the movement selection function, which, in this case, the custom variables are extracted from the
    # Kinarm data for each iteration.
    def custom_movement_selection_function(_trajectory):
        return movement_start_time, movement_end_time, movement_idx

    # create a Trajectory2D object to store the movement trajectory for the current trial
    trajectory = Trajectory2D(
        x=right_pos[:, 0],
        y=right_pos[:, 1],
        time=timestamps,

        vel_threshold=0.05,  # m/s, velocity threshold for movement onset detection

        displacement_preprocess=(Preprocesses.LOW_BUTTER, ),
        velocity_preprocess=(Preprocesses.CENT_DIFF, Preprocesses.LOW_BUTTER, ),
        acceleration_preprocess=(Preprocesses.CENT_DIFF, Preprocesses.LOW_BUTTER, ),

        movement_selection_ax='xy',  # the direction of the movement that is of interest
        custom_compute_movement_boundary=custom_movement_selection_function,
    )

    # we also want to figure out the number of velocity maxima during movement, which is indicative of the number of
    # corrective submovements. The movement velocity is the velocity along the movement direction and within the
    # movement boundaries
    temp_vel = trajectory.movement_velocity
    temp_vel_local_max = temp_vel[find_peaks(temp_vel)[0]]

    # remove the peak velocity from the local max (because the peak velocity is also a local max). Using a threshold
    # to remove the peak velocity because of floating point precision.
    temp_vel_local_max = temp_vel_local_max[np.abs(temp_vel_local_max - trajectory.peak_vel) > 0.001]

    temp_vel_local_max_df = pd.DataFrame({'n_local_max': len(temp_vel_local_max)}, index=[0])

    # append the kinematic data to the trial summary
    temp_summary = pd.concat([temp_summary, trajectory.format_results(), temp_vel_local_max_df], axis=1)

    # store the trial summary
    trial_summary = pd.concat([trial_summary, temp_summary], axis=0, ignore_index=True)

    # # uncomment to check for movement trajectories for each trial
    # trajectory.debug_plots(fig=fig_traj, axs=ax_traj)
    # # plt.suptitle(f'{trajectory.time_to_peak_vel:.2f} ms', fontsize=20)
    # # plt.suptitle(f'n local max = {len(temp_vel_local_max)}', fontsize=20)
    #
    # plt.pause(.5)
    # input('Hit enter')
    # [a.cla() for a in ax_traj.flatten()]

    # add the trajectory to the TrajectoryMean object
    trajectory_mean_all[trial_type_idx].add_trajectory(trajectory)

trial_summary.to_csv('./demo/sample_kinarm_analysis_results.csv', index=False)

fig_traj, ax_traj = plt.subplots(1, 1)
for idx, _ in enumerate(trial_type_all):
    # compute the mean trajectory for the current condition
    # NOTE: The default compute mean trajectory takes all three axis. However, because we only use two axes, we
    # need to manually specify the axes to compute the mean trajectory.
    trajectory_mean_all[idx].compute_mean_trajectory(traj_names=('x_fit', 'y_fit'))

    # plot the mean trajectory
    trajectory_mean_all[idx].debug_plots_trajectory(principal_dir='xy', fig=fig_traj, ax=ax_traj, show_text=True)

