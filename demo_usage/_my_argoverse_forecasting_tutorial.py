from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# %%

# set root_dir to the correct path to your dataset folder
root_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-forecasting/forecasting_train/sample/'

afl = ArgoverseForecastingLoader(root_dir)

print('Total number of sequences:', len(afl))
print(afl[4])

# %%
for argoverse_forecasting_data in (afl):
    print(argoverse_forecasting_data)

argoverse_forecasting_data = afl[0]
print(argoverse_forecasting_data.track_id_list)
print(argoverse_forecasting_data)
# %%
# seq_path = f"{root_dir}/2645.csv"
# viz_sequence(afl.get(seq_path).seq_df, show=True)
# seq_path = f"{root_dir}/3828.csv"
# viz_sequence(afl.get(seq_path).seq_df, show=True)

# %%

avm = ArgoverseMap()
# %%
def get_plot(map_range):
    my_dpi = 96.0
    # fig = plt.figure(figsize=(72 / my_dpi, 72 / my_dpi), dpi=my_dpi)
    fig = plt.figure(figsize=(5,5), dpi=my_dpi)

    # fig.tight_layout(pad=0)
    ax = fig.add_axes([0., 0., 1., 1.])
    ax.set_xlim([map_range[0], map_range[1]])
    ax.set_ylim([map_range[2], map_range[3]])
    # ax.axis('off')

    return fig, ax


def get_map_range(point):
    [xcenter, ycenter] = point
    r = 25
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
    ax.scatter(agent_obs_traj[:,0], agent_obs_traj[:,1], c='g', zorder=20)
    ax.scatter(agent_pred_traj[:,0], agent_pred_traj[:,1], c='b', zorder=20)
    return ax

obs_len = 20
csvs = ["2645.csv", "3700.csv", "3828.csv", "3861.csv", "4791.csv"]


seq_path = f"{root_dir}/2645.csv"

agent_obs_traj = afl.get(seq_path).agent_traj[:obs_len]
agent_pred_traj = afl.get(seq_path).agent_traj[obs_len:]
candidate_centerlines = avm.get_candidate_centerlines_for_traj(agent_obs_traj, afl[1].city, viz=True)

tot = 0
data = agent_obs_traj
for i in xrange(data.shape[0]-1):
    tot += ((((data[i+1:]-data[i])**2).sum(1))**.5).sum()

avg = tot/((data.shape[0]-1)*(data.shape[0])/2.)
city_name = afl[1].city
# print(city_name)
# print(agent_obs_traj[0])
center, map_range = get_map_range(agent_obs_traj[0])


fig, ax = get_plot(map_range)
ax = add_map(ax, map_range, city_name)
ax = add_trajectory(ax, agent_obs_traj, agent_pred_traj)
fig.show()


# seq_path = f"{root_dir}/3828.csv"
# agent_obs_traj = afl.get(seq_path).agent_traj[:obs_len]
# candidate_centerlines = avm.get_candidate_centerlines_for_traj(agent_obs_traj, afl[4].city, viz=True)
