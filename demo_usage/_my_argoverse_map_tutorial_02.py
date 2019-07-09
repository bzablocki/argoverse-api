# %%
from argoverse.map_representation.map_api import ArgoverseMap
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from visualize_30hz_benchmark_data_on_map import DatasetOnMapVisualizer
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from matplotlib.patches import Polygon
import pandas as pd
import pickle
import logging
import os

# %%
# tracking_dataset_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-tracking/sample/'
tracking_dataset_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-tracking/sample/'
# %%

# %%
am = ArgoverseMap()
# %%
log_index = 0
frame_index = 100
idx = 0
argoverse_loader = ArgoverseTrackingLoader(tracking_dataset_dir)
log_id = argoverse_loader.log_list[log_index]
argoverse_data = argoverse_loader[log_index]
city_name = argoverse_data.city_name

lidar_pts = argoverse_data.get_lidar(idx)
print(argoverse_data)
# %%
# Map from a bird's-eye-view (BEV)
dataset_dir = tracking_dataset_dir
experiment_prefix = 'visualization_demo'

# if you are running for the first time, or using a new set of logs, this will need to be set False to accumelate the labels again
use_existing_files = True

city_to_egovehicle_se3 = argoverse_data.get_pose(idx)

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

domv = DatasetOnMapVisualizer(dataset_dir, experiment_prefix,
                              use_existing_files=use_existing_files, log_id=argoverse_data.current_log)

# %% markdown
# One example is to overlay our label annotations on top of our map information. Here the pink area denotes the `driveable area`
# %%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
xcenter, ycenter, _ = argoverse_data.get_pose(idx).translation

r = 50
xmin = xcenter - r  # 150
xmax = xcenter + r  # 150
ymin = ycenter - r  # 150
ymax = ycenter + r  # 150
ax.scatter(xcenter, ycenter, 200, color="g", marker=".", zorder=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
local_lane_polygons = am.find_local_lane_polygons([xmin, xmax, ymin, ymax], city_name)
local_das = am.find_local_driveable_areas([xmin, xmax, ymin, ymax], city_name)


domv.render_bev_labels_mpl(
    city_name=city_name,
    ax=ax,
    axis="city_axis",
    lidar_pts=None,
    local_lane_polygons=copy.deepcopy(local_lane_polygons),
    local_das=copy.deepcopy(local_das),
    log_id=log_id,
    timestamp=argoverse_data.lidar_timestamp_list[idx],
    city_to_egovehicle_se3=city_to_egovehicle_se3,
    avm=am,
    vis_other_objects=True
)

prev_timestamp = argoverse_data.lidar_timestamp_list[idx]
print("timestamp: {}".format(prev_timestamp))

# %%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
idx_new = idx + 1
xcenter, ycenter, _ = argoverse_data.get_pose(idx_new).translation
ax.scatter(xcenter, ycenter, 200, color="g", marker=".", zorder=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

domv.render_bev_labels_mpl(
    city_name=city_name,
    ax=ax,
    axis="city_axis",
    lidar_pts=None,
    local_lane_polygons=copy.deepcopy(local_lane_polygons),
    local_das=copy.deepcopy(local_das),
    log_id=log_id,
    timestamp=argoverse_data.lidar_timestamp_list[idx_new],
    city_to_egovehicle_se3=city_to_egovehicle_se3,
    avm=am,
    vis_other_objects=True
)

new_timestamp = argoverse_data.lidar_timestamp_list[idx_new]
print("timestamp diff: {}".format((new_timestamp - prev_timestamp)))

# %%
objects = domv.log_timestamp_dict[log_id][argoverse_data.lidar_timestamp_list[idx + 10]]
len(objects), objects[1].track_uuid
# for i, obj in enumerate(objects):
#     print(np.linalg.norm(argoverse_data.get_pose(idx).translation - np.mean(obj.bbox_city_fr, axis=0)))

for i, obj in enumerate(objects):
    print(obj.bbox_city_fr)
# %%

# %%

# %%
idx = 100

fig, ax = plt.subplots(figsize=(10, 10))
rrr = 0
ax.set_xlim([xmin - rrr, xmax + rrr])
ax.set_ylim([ymin - rrr, ymax + rrr])

# create map
poly = local_das[0]
ax.add_patch(Polygon(poly[:, 0:2], facecolor="white", alpha=1))
for i in range(1, len(local_das)):
    poly = local_das[i]
    ax.add_patch(Polygon(poly[:, 0:2], facecolor="black", alpha=1))

# display ego
xcenter, ycenter, _ = argoverse_data.get_pose(idx).translation
ax.scatter(xcenter, ycenter, 300, color="g", marker=".", zorder=2)

# display other objects
objects = domv.log_timestamp_dict[log_id][argoverse_data.lidar_timestamp_list[idx]]
for i, obj in enumerate(objects):
    if obj.obj_class_str == "VEHICLE":
        ax.scatter(np.mean(obj.bbox_city_fr, axis=0)[0], np.mean(obj.bbox_city_fr, axis=0)[
                   1], 200, color="r", marker=".", zorder=2, alpha=0.2)

objects = domv.log_timestamp_dict[log_id][argoverse_data.lidar_timestamp_list[idx + 15]]
for i, obj in enumerate(objects):
    if obj.obj_class_str == "VEHICLE":
        ax.scatter(np.mean(obj.bbox_city_fr, axis=0)[0], np.mean(obj.bbox_city_fr, axis=0)[
                   1], 200, color="r", marker=".", zorder=2, alpha=0.4)

objects = domv.log_timestamp_dict[log_id][argoverse_data.lidar_timestamp_list[idx + 30]]
for i, obj in enumerate(objects):
    if obj.obj_class_str == "VEHICLE":
        ax.scatter(np.mean(obj.bbox_city_fr, axis=0)[0], np.mean(obj.bbox_city_fr, axis=0)[
                   1], 200, color="r", marker=".", zorder=2, alpha=0.8)


plt.show()
# %%
idxx = 0
objects = domv.log_timestamp_dict[log_id][argoverse_data.lidar_timestamp_list[idxx]]
uuid = objects[4].track_uuid
# %%
# unique_objects = set()
objects_from_to = dict()

for idxx in range(len(argoverse_data.lidar_timestamp_list)):
    objects = domv.log_timestamp_dict[log_id][argoverse_data.lidar_timestamp_list[idxx]]
    for i, obj in enumerate(objects):
        if obj.obj_class_str == "VEHICLE":
            #   unique_objects.add(obj.track_uuid)
            if obj.track_uuid not in objects_from_to:
                objects_from_to[obj.track_uuid] = dict()
                objects_from_to[obj.track_uuid]['start'] = idxx
                objects_from_to[obj.track_uuid]['positions10Hz'] = []
                objects_from_to[obj.track_uuid]['positions10Hz'].append(
                    np.concatenate(([idxx], np.mean(obj.bbox_city_fr, axis=0))))
            else:
                objects_from_to[obj.track_uuid]['stop'] = idxx
                objects_from_to[obj.track_uuid]['positions10Hz'].append(
                    np.concatenate(([idxx], np.mean(obj.bbox_city_fr, axis=0))))


print(len(objects_from_to))

# %%


def get_plot(map_range, pix_to_pix_mapping=True):

    if pix_to_pix_mapping:
        my_dpi = 96.0  # screen constant, check here https://www.infobyip.com/detectmonitordpi.php
        fig = plt.figure(figsize=((map_range[1] - map_range[0]) / my_dpi,
                                  (map_range[3] - map_range[2]) / my_dpi), dpi=my_dpi)
    else:
        fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111)
    ax.set_xlim([map_range[0], map_range[1]])
    ax.set_ylim([map_range[2], map_range[3]])
    ax.axis('off')

    return fig, ax


def add_map(ax, map_range):
    city_name = argoverse_data.city_name
    map_polygons = am.find_local_driveable_areas(map_range, city_name)

    poly = map_polygons[0]
    ax.add_patch(Polygon(poly[:, 0:2], facecolor="white", alpha=1))
    for i in range(1, len(map_polygons)):
        poly = map_polygons[i]
        ax.add_patch(Polygon(poly[:, 0:2], facecolor="black", alpha=1))

    return ax


def add_ego(ax, ego_pos):
    ax.scatter(ego_pos[0], ego_pos[1], 300, color="g", marker=".", zorder=2)
    return ax


def add_vehicle_path(ax, positions):
    positions = np.array(positions)
    ax.scatter(positions[:, 1], positions[:, 2], 100, color="r", marker=".", zorder=2, alpha=0.2)

    return ax


def add_other_vehicles_paths(ax, other_vehicles):
    for veh in other_vehicles.values():
        positions = np.array(veh['positions10Hz'])
        ax.scatter(positions[:, 1], positions[:, 2], 100, color="blue", marker=".", zorder=2, alpha=0.2)
    return ax


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def visualize_vehicle_path(uuid, target_object, viz_trajectories=False):
    idx_start = target_object['start']
    idx_stop = target_object['stop']
    positions = target_object['positions10Hz']
    map_range = target_object['map_range']
    ego_pos = target_object['ego_pos']
    other_vehicles = target_object['other_vehicles']

    print(f"Actual path of {uuid}, start: {idx_start}, stop: {idx_stop}, length: {idx_stop - idx_start}")

    fig, ax = get_plot(map_range, pix_to_pix_mapping=True)
    ax = add_map(ax, map_range)
    if viz_trajectories:
        ax = add_ego(ax, ego_pos)
        ax = add_vehicle_path(ax, positions)
        ax = add_other_vehicles_paths(ax, other_vehicles)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = rgb2gray(data)

    target_object['img'] = data

#     plt.show()


# %%
def get_map_range(idx):
    xcenter, ycenter, _ = argoverse_data.get_pose(idx).translation
    r = 25
    xmin = xcenter - r
    xmax = xcenter + r
    ymin = ycenter - r
    ymax = ycenter + r

    return [xcenter, ycenter], [xmin, xmax, ymin, ymax]


def interpolate_missing_frames(positions):
    interpolated_positions = []
    idxs = positions[:, 0]
    pos_x = positions[:, 1]
    pos_y = positions[:, 2]

    desired_idxs = np.arange(idxs[0], idxs[-1] + 1, 1)

    new_pos_x = np.interp(desired_idxs, idxs, pos_x)
    new_pos_y = np.interp(desired_idxs, idxs, pos_y)

    interpolated_positions = np.column_stack((desired_idxs, new_pos_x, new_pos_y))
    return interpolated_positions


def is_within(pos, map_range):
    [xmin, xmax, ymin, ymax] = map_range
    return xmin <= pos[1] <= xmax and ymin <= pos[2] <= ymax


def trim_path_to_visible(positions):
    valid_positions = []
    positions = np.array(positions)

    map_range = None
    ego_pos = None
    starting_position_found = False

    for pos in positions:
        if not starting_position_found:
            ego_pos, map_range = get_map_range(int(pos[0]))

        if is_within(pos, map_range):
            valid_positions.append(pos)
            starting_position_found = True

    return ego_pos, map_range, np.array(valid_positions)


def trim_and_interpolate_object(target_object):
    positions = target_object['positions10Hz']
    ego_pos, map_range, positions = trim_path_to_visible(positions)

    if positions.size:
        positions = interpolate_missing_frames(positions)

        target_object['start'] = int(positions[0][0])
        target_object['stop'] = int(positions[-1][0])
        target_object['positions10Hz'] = positions
        target_object['map_range'] = map_range
        target_object['ego_pos'] = ego_pos

        return target_object

    return None


def add_other_vehicles(target_uuid, target_object, other_objects):
    # we're looking for other vehicles around our target vehicles, to model social forces
    map_range = target_object['map_range']
    target_start = target_object['start']
    target_stop = target_object['stop']

    other_objects_valid = dict()

    # process every other vehicle one by one and check if it's visible by ego vehicle and add to other vehicles data structure
    for other_uuid, other_object in other_objects.items():
        # don't compare to self
        if target_uuid == other_uuid:
            pass
        else:
            other_postitions = np.array(other_object['positions10Hz'])
            # get only the positions in the timeframe of the target vehicle
            other_postitions = other_postitions[(other_postitions[:, 0] >= target_start)
                                                & (other_postitions[:, 0] <= target_stop)]

            # iterate through all the positions
            for other_pos in other_postitions:

                # check if ego_vehicle can actually see it
                if is_within(other_pos, map_range):
                    # if the dict doesn't have an entry with the uuid, initialize it
                    if other_uuid not in other_objects_valid:
                        other_objects_valid[other_uuid] = dict()
                        other_objects_valid[other_uuid]['positions10Hz'] = None

                    # add the position entry if the ego vehicle can see the object
                    other_objects_valid[other_uuid]['positions10Hz'] = np.vstack(
                        (other_objects_valid[other_uuid]['positions10Hz'], other_pos)) if other_objects_valid[other_uuid]['positions10Hz'] is not None else np.reshape(other_pos, (1, -1))

    target_object['other_vehicles'] = other_objects_valid
    return target_object


def add_img(new_target_object):
    map_range = target_object['map_range']

    fig, ax = get_plot(map_range, pix_to_pix_mapping=True)
    ax = add_map(ax, map_range)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     data = rgb2gray(data)

    target_object['img'] = data
    target_object['img_shape'] = data.shape
    plt.close()

    return target_object


valid_target_objects = dict()
for i, (uuid, target_object) in enumerate(objects_from_to.items()):
    # get a segment of the path that is visible by ego vehicle, and interpolate missing frames
    new_target_object = trim_and_interpolate_object(target_object)
    if new_target_object is not None:
        new_target_object = add_other_vehicles(uuid, new_target_object, objects_from_to)
        new_target_object = add_img(new_target_object)
        valid_target_objects[uuid] = new_target_object

print(len(valid_target_objects))
uuid = list(valid_target_objects.keys())[0]
print(uuid)
# valid_target_objects[uuid]['other_vehicles']['605e47ae-13a8-478e-9edc-3175ae125908']['positions10Hz']
# %%
for i in range(0, 1):
    uuid = list(valid_target_objects.keys())[i]
    target_object = valid_target_objects[uuid]
    visualize_vehicle_path(uuid, target_object)

# %%

# %%


def save_to_pickle():
    pickle_out = open("/media/bartosz/hdd1TB/workspace_hdd/SS-LSTM/data/argoverse/train_data_01.pickle", "wb")
    pickle.dump(valid_target_objects, pickle_out)
    pickle_out.close()


# save_to_pickle()
# %%
print(valid_target_objects['d9947a79-b94f-4db9-9f79-7b7aeb803354']['other_vehicles']
      ['ef1474cf-ab4c-4b10-b366-d1ddf8e3b632']['positions10Hz'].shape)

# %%
a = np.array([1, 2, 3])
a.shape

b = np.array([4, 5, 6])
ab = np.vstack((a, b))
print(ab.shape)

c = np.array([4, 5, 6])
