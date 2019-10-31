from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
# %%

# set root_dir to the correct path to your dataset folder
# root_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-forecasting/forecasting_train/sample/'
root_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-forecasting/train/data'

afl = ArgoverseForecastingLoader(root_dir)

print('Total number of sequences:', len(afl))

avm = ArgoverseMap()
# %%


def get_plot(map_range):
    my_dpi = 96.0
    # fig = plt.figure(figsize=(72 / my_dpi, 72 / my_dpi), dpi=my_dpi)
    fig = plt.figure(figsize=(5, 5), dpi=my_dpi)

    # fig.tight_layout(pad=0)
    ax = fig.add_axes([0., 0., 1., 1.])
    ax.set_xlim([map_range[0], map_range[1]])
    ax.set_ylim([map_range[2], map_range[3]])
    # ax.axis('off')

    return fig, ax


def get_map_range(point, r=25):
    [xcenter, ycenter] = point
    xmin = xcenter - r
    xmax = xcenter + r
    ymin = ycenter - r
    ymax = ycenter + r

    return [xcenter, ycenter], [xmin, xmax, ymin, ymax]


def add_map(ax, map_range, city_name):
    map_polygons = avm.find_local_lane_polygons(map_range, city_name)

    # ax.set_facecolor("black")
    for i in range(0, len(map_polygons)):
        poly = map_polygons[i]
        ax.add_patch(Polygon(poly[:, 0:2], facecolor="black", alpha=1))

    return ax



def add_trajectory(ax, agent_obs_traj, agent_pred_traj):
    ax.scatter(agent_obs_traj[:, 0], agent_obs_traj[:, 1], c='g', zorder=20)
    ax.scatter(agent_pred_traj[:, 0], agent_pred_traj[:, 1], c='b', zorder=20)
    return ax

def calculate_mean_std():
    total_avg = []
    obs_len = 20
    for idx in range(0, 10):
        traj = afl[idx].agent_traj
        agent_obs_traj = traj[:obs_len]
        agent_pred_traj = traj[obs_len:]
        candidate_centerlines = avm.get_candidate_centerlines_for_traj(agent_obs_traj, afl[idx].city, viz=False)

        distances = []
        data = traj

        for i in range(data.shape[0] - 1):
            distances.append(np.linalg.norm(data[i] - data[i + 1], axis=0))
        total_avg.append(np.mean(distances))

        city_name = afl[idx].city
        center, map_range = get_map_range(agent_obs_traj[-1], r=40)

        fig, ax = get_plot(map_range)
        ax = add_map(ax, map_range, city_name)
        ax = add_trajectory(ax, agent_obs_traj, agent_pred_traj)
        ax.set_title(np.mean(distances))
        fig.show()

    print(f"Final avg {np.sum(total_avg)/len(total_avg)}, std: {np.std(total_avg)}, max: {np.max(total_avg)}")
# calculate_mean_std()

def vizualize_sample():
    obs_len = 20
    # for idx in range(10, len(afl)):
    for idx in range(10, 30):
        # if idx > 20:
        #     break
        traj = afl[idx].agent_traj
        agent_obs_traj = traj[:obs_len]
        agent_pred_traj = traj[obs_len:]
        candidate_centerlines = avm.get_candidate_centerlines_for_traj(agent_obs_traj, afl[idx].city, viz=False)
        city_name = afl[idx].city
        center, map_range = get_map_range(agent_obs_traj[-1], r=40)

        fig, ax = get_plot(map_range)
        ax = add_map(ax, map_range, city_name)
        ax = add_trajectory(ax, agent_obs_traj, agent_pred_traj)
        ax.set_title(str(idx))
        fig.show()
# vizualize_sample()




# seq_path = f"{root_dir}/3828.csv"
# agent_obs_traj = afl.get(seq_path).agent_traj[:obs_len]
# candidate_centerlines = avm.get_candidate_centerlines_for_traj(agent_obs_traj, afl[4].city, viz=True)
