"""
Sample data visualization script for the spatial cueing experiment in Wang & Welsh (2023). Written by X.M. Wang.

Wang, X.M., & Welsh, T.N. (2023). TAT-HUM: Trajectory Analysis Toolkit for Human Movements in Python.

"""

# %% import the relevant libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% plot MT/RT
raw_data = pd.read_csv('./demo/rt_mt_results.csv')

def replace_congruency_label(label):
    return 'Cued' if label == 'congruent' else 'Uncued'

raw_data['congruency'] = raw_data['congruency'].apply(lambda x: replace_congruency_label(x))

ax_label_size = 20
ax_tick_size = 14
line_width = 3

# toggle this to check which dependent variable to plot
# plot_dv = 'rt'
plot_dv = 'mt'

fig = plt.figure()
ax = sns.lineplot(
    data=raw_data,

    x='soa',
    y=plot_dv,
    hue='congruency',
    style='congruency',

    palette=sns.color_palette('gray')[:2],  # two congruencies, congruent and incongruent

    linewidth=line_width,

    err_kws={'capsize': 10, 'capthick': line_width, 'linewidth': line_width},
    err_style='bars',
)

ax.legend().set_title('')
if plot_dv == 'rt':
    ax.set_ylabel('Reaction Time (ms)', fontsize=ax_label_size)

    ax.text(63, 454, '***', fontsize=ax_label_size)
    ax.text(1066, 389, '**', fontsize=ax_label_size)

    ax.set_ylim((340, 470))
else:
    ax.set_ylim((300, 400))
    ax.set_ylabel('Movement Time (ms)', fontsize=ax_label_size)

ax.set_xlabel('SOA (ms)', fontsize=ax_label_size)
ax.set_xticks([100, 350, 850, 1100])
ax.set_xticklabels([100, 350, 850, 1100], fontsize=ax_tick_size)

ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=ax_tick_size)
plt.legend(fontsize=ax_label_size)
plt.subplots_adjust(bottom=0.13, left=0.16, top=0.95, right=0.95)
plt.show()

fig.savefig(f'./demo/{plot_dv}.png')  # uncomment this to automatically save the figure

# %% plot congruency area

mean_area = pd.read_csv('./demo/mean_area_results.csv')

ax_label_size = 14
ax_tick_size = 11
line_width = 2

x_tick_labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
par_id_all = np.sort(mean_area['par_id'].unique())
print(f'{len(par_id_all)} participants: {par_id_all}')
soa_all = np.sort(mean_area['soa'].unique())
target_all = mean_area['target_location'].unique()
sub_ind_all = mean_area['sub_ind'].unique()

line_styles = {'color': ['k', 'k', 'k', 'k'], "ls": ["-", "--", ':', '-.']}
g = sns.FacetGrid(
    mean_area,

    col='soa',
    hue='soa',
    hue_kws=line_styles,
    # row='par_id',

    margin_titles=True,
)

g.map(
    sns.lineplot,

    'sub_ind',
    'sub_area_continuous',

    linewidth=line_width,
    err_style='bars',
    err_kws={'capsize': 10, 'capthick': line_width, 'linewidth': line_width},
)

g.set_axis_labels('', '')
g.fig.supxlabel('Percentage of Trajectory (%)', fontsize=ax_label_size)
g.fig.supylabel(r'Trajectory Area ($mm^{2}$)', fontsize=ax_label_size)

# g.set(ylim=(-40, 40), )
for ax1 in g.axes:
    for ax2 in ax1:
        ax2.plot([19, 99], [0, 0], 'k:', linewidth=3)
        ax2.set_xticks([20, 40, 60, 80, 100])
        ax2.set_yticklabels(ax2.get_yticks().astype(int), fontsize=ax_tick_size)
        ax2.set_title(ax2.get_title().upper().replace('.0', ' ms'))

g.axes[0, 1].text(37., -7.5, '*', fontsize=ax_label_size)
g.axes[0, 3].text(37., -7.5, '*', fontsize=ax_label_size)
g.axes[0, 3].text(56.5, -11, '*', fontsize=ax_label_size)
g.axes[0, 3].text(75, -10, '**', fontsize=ax_label_size)
g.axes[0, 3].text(93, -7, '***', fontsize=ax_label_size)

g.set_xticklabels(x_tick_labels, fontsize=ax_tick_size)
plt.subplots_adjust(left=0.09, wspace=0.18)
plt.show()

g.fig.savefig('./demo/TrajectoryArea.png')  # uncomment this to automatically save the figure
