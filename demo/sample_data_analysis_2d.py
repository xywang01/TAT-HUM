import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tathum.functions import Preprocesses
from tathum.trajectory_2d import Trajectory2D

raw_data = pd.read_csv('./demo/demo_data/demo_data_2d.csv')
transform_data = pd.read_csv('./demo/demo_data/transform_data_2d.csv')

par_id_all = raw_data['par_id'].unique()
vib_all = raw_data['vib'].unique()
modality_all = raw_data['modality'].unique()
vision_all = raw_data['vision'].unique()

for par_id in par_id_all:
    for modality in modality_all:
        temp_transform = transform_data[
            (transform_data['par_id'] == par_id) &
            (transform_data['modality'] == modality)]
        exp_end_point = temp_transform[['x', 'y']].to_numpy()
        exp_end_point = (exp_end_point[0], exp_end_point[1])  # reformat the data for the appropriate input type

        for vib in vib_all:
            for vision in vision_all:
                df_idx = np.where((raw_data['par_id'] == par_id) &
                                  (raw_data['vib'] == vib) &
                                  (raw_data['modality'] == modality) &
                                  (raw_data['vision'] == vision))[0]
                temp = raw_data.iloc[df_idx]

                temp_x = temp['x'].values
                temp_y = temp['y'].values

                temp_x[[2, 3, 4, 6, 7]] = 0.
                temp_y[[2, 3, 4, 6, 7]] = 0.

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

                plt.figure()

                plt.scatter(traj.x, traj.y)
                plt.plot(traj.x_fit, traj.y_fit)

                # plt.plot(traj.transform_origin[0], traj.transform_origin[1], marker='o', color='black')
                # x_rot, y_rot = traj.transform_data(traj.x, traj.y)
                # plt.plot(x_rot, y_rot)
                plt.axis('equal')
                raise ValueError



