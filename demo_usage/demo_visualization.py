import argoverse.visualization.visualization_utils as viz_util
from argoverse.utils.frustum_clipping import generate_frustum_planes
from PIL import Image
from cuboids_to_bboxes import plot_lane_centerlines_in_img
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import logging
import matplotlib.pyplot as plt
from visualize_30hz_benchmark_data_on_map import DatasetOnMapVisualizer
from argoverse.map_representation.map_api import ArgoverseMap
import os
import copy
from matplotlib.patches import Polygon
import numpy as np
from argoverse.utils.geometry import filter_point_cloud_to_polygon, rotate_polygon_about_pt
from argoverse.utils.mpl_plotting_utils import draw_lane_polygons, plot_bbox_2D
import matplotlib.patches as patches
import pickle
from skimage.transform import resize
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, AnnotationBbox)
from matplotlib.cbook import get_sample_data
import cv2
import scipy.misc
import png
import matplotlib.image as mpimg

IS_OCCLUDED_FLAG = 100


logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# train4 log = 2, idx = 60 from straight, going straight on intersection
# train3 log = 13, idx = 30 -> 130 from left, going straight on intersection
# train3 log = 14, idx = 0 -> 70 from straight, going right on intersection

obs_frames, pred_frames = 20, 50
visualization_dir = "/media/bartosz/hdd1TB/workspace_hdd/argoverse-api/demo_usage/my_vizualization"

# tracking_dataset_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-tracking/train4/'
tracking_dataset_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-tracking/train3/'
argoverse_map = ArgoverseMap()
# Map from a bird's-eye-view (BEV)
dataset_dir = tracking_dataset_dir
experiment_prefix = 'visualization_demo'
log_index = 14  # 14
argoverse_loader = ArgoverseTrackingLoader(tracking_dataset_dir)
log_id = argoverse_loader.log_list[log_index]
argoverse_data = argoverse_loader[log_index]
city_name = argoverse_data.city_name

use_existing_files = True
domv = DatasetOnMapVisualizer(dataset_dir, experiment_prefix, use_existing_files=use_existing_files, log_id=argoverse_data.current_log)

# %%
camera = 'ring_front_center'
# camera = 'ring_front_left'
calib = argoverse_data.get_calibration(camera)
planes = generate_frustum_planes(calib.camera_config.intrinsic.copy(), camera)

domv = DatasetOnMapVisualizer(tracking_dataset_dir, experiment_prefix, use_existing_files=True, log_id=argoverse_data.current_log)
# %%


def get_plot(map_range, axis_off=True, for_network=False):

    my_dpi = 96.0  # screen constant, check here https://www.infobyip.com/detectmonitordpi.php
    if for_network:
        fig = plt.figure(figsize=(72 / my_dpi, 72 / my_dpi), dpi=my_dpi)
    else:
        fig = plt.figure(figsize=(496 / my_dpi, 496 / my_dpi), dpi=my_dpi)

    # fig.tight_layout(pad=0)
    ax = fig.add_axes([0., 0., 1., 1.])
    ax.set_xlim([map_range[0], map_range[1]])
    ax.set_ylim([map_range[2], map_range[3]])
    if axis_off:
        ax.axis('off')

    return fig, ax


def get_lanes_before_after(lane_candidates):

    lanes_around = []
    for lane in lane_candidates:
        dfs_threshold = 2
        candidates_future = argoverse_map.dfs(lane, city_name, 0, dfs_threshold)
        candidates_past = argoverse_map.dfs(lane, city_name, 0, dfs_threshold, True)
        # Merge past and future
        for past_lane_seq in candidates_past:
            for future_lane_seq in candidates_future:
                assert past_lane_seq[-1] == future_lane_seq[0], "Incorrect DFS for candidate lanes past and future"
                lanes_around.append(past_lane_seq + future_lane_seq[1:])

    return lanes_around


def get_lanes(xcenter, ycenter, r, city_name, curr_lane_candidates=None):
    if curr_lane_candidates is None:
        curr_lane_candidates = argoverse_map.get_lane_ids_in_xy_bbox(xcenter, ycenter, city_name, r)
    lanes_around = get_lanes_before_after(curr_lane_candidates)
    candidate_cl = argoverse_map.get_cl_from_lane_seq(lanes_around, city_name)
    trajectories = []
    for line_nb, centerline_coords in enumerate(candidate_cl):
        print(centerline_coords)
        line_coords = list(zip(*centerline_coords))
        trajectories.append(line_coords)
    return trajectories


def add_other_vehicles(ax, log_id, timestamp, city_to_egovehicle_se3, map_range):
    objects = domv.log_timestamp_dict[log_id][timestamp]

    all_occluded = True
    for frame_rec in objects:
        if frame_rec.occlusion_val != IS_OCCLUDED_FLAG:
            all_occluded = False

    if not all_occluded:
        for i, frame_rec in enumerate(objects):
            bbox_city_fr = frame_rec.bbox_city_fr
            bbox_ego_frame = frame_rec.bbox_ego_frame
            color = frame_rec.color

            if frame_rec.occlusion_val != IS_OCCLUDED_FLAG:
                bbox_ego_frame = rotate_polygon_about_pt(bbox_ego_frame, city_to_egovehicle_se3.rotation, np.zeros((3,)))
                # plot_bbox_2D(ax, bbox_city_fr, "red")
                # xt, yt = bbox_city_fr[0,0]+20, bbox_city_fr[0,1]-5
                # target_vehicle_uuid = "959b6730-fd74-4569-b86a-2363233b48bd"
                # if "48bd" in frame_rec.track_uuid :
                # if frame_rec.track_uuid == target_vehicle_uuid:
                #     ax.set_title(str(np.mean(bbox_city_fr[:, 0])) + " " + str(np.mean(bbox_city_fr[:, 1])))
                #     print(frame_rec.track_uuid)

                # if xt > map_range[0] and xt < map_range[1] and yt > map_range[2] and yt < map_range[3]:
                #     ax.text(xt, yt, frame_rec.track_uuid)
                ax.scatter(np.mean(bbox_city_fr[:, 0]), np.mean(bbox_city_fr[:, 1]), 600, color="r", marker=".", zorder=500)

    return ax


def add_lanes(xcenter, ycenter, r, city_name):
    lanes = get_lanes(xcenter, ycenter, r + 50, city_name)
    for lane in lanes:
        ax.plot(lane[0], lane[1], c='g')

    return ax


def add_map(ax, map_range, argoverse_data, argoverse_map, color="black"):
    city_name = argoverse_data.city_name
    map_polygons = argoverse_map.find_local_lane_polygons(map_range, city_name)

    # ax.set_facecolor("black")
    for i in range(0, len(map_polygons)):
        poly = map_polygons[i]
        ax.add_patch(Polygon(poly[:, 0:2], facecolor="black", alpha=1))

    return ax


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def save_map(fig, ax, idx, log_index, save_jpg_npy, dir=None):
    ax.axis('off')

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if save_jpg_npy == "npy":
        data = np.rot90(data.transpose(1, 0, 2))
        data = 1 - rgb2gray(data)

        path = os.path.join(visualization_dir, "map_network", "{}_{}.npy".format(log_index, idx))
        np.save(path, data)
    if save_jpg_npy == "jpg":
        # print(data.shape)
        im = Image.fromarray(data)
        # im =  im.convert('RGB')
        # display(im)
        dir = dir if dir is not None else "map_lanes"
        path = os.path.join(visualization_dir, dir, "{}_{}.jpg".format(log_index, idx))
        im.save(path)
        print(f"Save to {path}")


def add_self(ax, xcenter, ycenter):
    ax.scatter(xcenter, ycenter, 600, color="b", marker=".", zorder=5)
    # w, h = 4, 1.5
    # rect = patches.Rectangle((xcenter - (w / 2), ycenter - (h / 2)), w, h, linewidth=1, edgecolor='blue', facecolor='blue', zorder=100)
    # ax.add_patch(rect)
    return ax


def normalize(x, y, map_range):
    [xmin, xmax, ymin, ymax] = map_range
    x_c, y_c = np.copy(x), np.copy(y)
    x_c = (x_c - xmin) / (xmax - xmin)
    y_c = (y_c - ymin) / (ymax - ymin)
    return x_c, y_c


def get_target_vehicle_position(idx, log_id, map_range, prev_object, city_to_egovehicle_se3, is_normalized=True):
    target_vehicle_uuid = "959b6730-fd74-4569-b86a-2363233b48bd"
    timestamp = argoverse_data.lidar_timestamp_list[idx]
    objects = domv.log_timestamp_dict[log_id][timestamp]
    target_object = None
    for object in objects:
        if object.track_uuid == target_vehicle_uuid:
            target_object = object
            break

    [xmin, xmax, ymin, ymax] = map_range

    # let's just assume that prev object is not None
    if target_object is None:
        x, y = 750.3311659007195, 1533.1006558539234
        if is_normalized:
            x = (x - xmin) / (xmax - xmin)
            y = (y - ymin) / (ymax - ymin)
        return np.array([[x, y]]), prev_object

    bbox_city_fr = target_object.bbox_city_fr
    x, y = np.mean(bbox_city_fr[:, 0]), np.mean(bbox_city_fr[:, 1])
    if is_normalized:
        x, y = normalize(x, y, map_range)
        # x = (x - xmin) / (xmax - xmin)
        # y = (y - ymin) / (ymax - ymin)
    return np.array([[x, y]]), target_object


def add_target_lanes(ax, idx, log_id, map_range, city_name):
    city_to_egovehicle_se3 = argoverse_data.get_pose(idx)
    position, _ = get_target_vehicle_position(idx, log_id, map_range, None, city_to_egovehicle_se3, is_normalized=False)

    # curr_lane_candidates = argoverse_map.get_lane_ids_in_xy_bbox(position[0, 0], position[0, 1], city_name, 0.5)
    # print(f"curr_lane_candidates {curr_lane_candidates}")
    lanes = get_lanes(position[0, 0], position[0, 1], 1, city_name, curr_lane_candidates=[9620341, 9621010, 9629471])
    for lane in lanes:
        ax.plot(lane[0], lane[1], c='g')

    return ax


def vis_predictions(ax, idx, predictions_grid):
    predictions_grid = predictions_grid[idx]

    # Annotate the 2nd position with another image (a Grace Hopper portrait)
    # fn = get_sample_data("grace_hopper.png", asfileobj=False)
    # arr_img = resize(predictions_grid, (496.0, 496.0), anti_aliasing=False, order=0)

    with get_sample_data("/media/bartosz/hdd1TB/workspace_hdd/argoverse-api/demo_usage/my_vizualization/sslstm_predictions/01.jpg") as file:
        arr_img = plt.imread(file, format='jpg')
    xy = [720, 1540]
    imagebox = OffsetImage(arr_img, zoom=0.2)
    imagebox.image.axes = ax

    ab = AnnotationBbox(imagebox, xy,
                        xybox=(120., -80.),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.5,
                        arrowprops=dict(
                            arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=90,rad=3")
                        )

    ax.add_artist(ab)

    # ax.imshow(resize(predictions_grid, (496.0, 496.0), anti_aliasing=False, order=0), cmap='gray', alpha=1)

    return ax


def generate_network_input_data(preview=False, save_to_pickle=False):
    person_input = np.zeros((0, obs_frames, 2))
    expected_output = np.zeros((0, pred_frames, 2))
    scene_input = np.zeros((0, 1))
    social_input = np.zeros((0, obs_frames, 64))

    ref_frame = 5
    xcenter, ycenter, _ = argoverse_data.get_pose(ref_frame).translation
    r = 25
    ref_map_range = [xcenter - 5, xcenter + r + 20, ycenter - r, ycenter + r]
    ref_map_path = os.path.join(visualization_dir, "map_network", "{}_{}.npy".format(log_index, ref_frame))
    ref_city_to_egovehicle_se3 = argoverse_data.get_pose(ref_frame)

    for ref_frame in range(0, 70):
        prev_object = None
        positions = np.zeros((0, 2))
        map_paths = np.zeros((0, 1))

        for idx in range(ref_frame, ref_frame + obs_frames + pred_frames, 1):
            position, prev_object = get_target_vehicle_position(idx, log_id, ref_map_range, prev_object, ref_city_to_egovehicle_se3)
            positions = np.vstack((positions, position))

        obs = positions[0:obs_frames]
        pred = positions[obs_frames:obs_frames + pred_frames]

        if preview:
            ref_img = np.load(ref_map_path)
            plt.imshow(ref_img)
            # positions = positions * 72.0
            plt.scatter(obs[:, 0] * 72.0, obs[:, 1] * 72.0)
            plt.scatter(pred[:, 0] * 72.0, pred[:, 1] * 72.0)
            # plt.scatter(positions[:, 0], positions[:, 1])
            plt.show()

        person_input = np.vstack((person_input, np.expand_dims(obs, 0)))
        expected_output = np.vstack((expected_output, np.expand_dims(pred, 0)))
        scene_input = np.vstack((scene_input, np.expand_dims(ref_map_path, 0)))
        social_input = np.vstack((social_input, np.zeros((1, obs_frames, 64))))

    if save_to_pickle:
        print(person_input.shape, expected_output.shape, scene_input.shape, social_input.shape)
        name = os.path.join(visualization_dir, "network_data", "visualization_static_scene02.pickle")
        if os.path.exists(name):
            print(f"File {name} exists! Aborting.")
        else:
            pickle_out = open(name, "wb")
            pickle.dump([scene_input, social_input, person_input, expected_output], pickle_out, protocol=2)
            pickle_out.close()
            print("Saved to pickle {}".format(name))

    return scene_input, social_input, person_input, expected_output

# _ = generate_network_input_data(preview=False, save_to_pickle=False)


def generate_map(idx, log_id, is_add_other_vehicles=True, is_add_all_lanes=False,
                 is_add_target_lanes=False, is_add_self=True, save=False, save_jpg_npy="jpg", dir=None):
    # top-down map view
    city_to_egovehicle_se3 = argoverse_data.get_pose(idx)
    xcenter, ycenter, _ = argoverse_data.get_pose(idx).translation
    r = 25
    map_range = [xcenter - 5, xcenter + r + 20, ycenter - r, ycenter + r]  # [xmin, xmax, ymin, ymax]

    fig, ax = get_plot(map_range, axis_off=False, for_network=False)
    ax = add_map(ax, map_range, argoverse_data, argoverse_map, color="black")  # lightgrey
    if is_add_other_vehicles:
        ax = add_other_vehicles(ax, log_id, argoverse_data.lidar_timestamp_list[idx], city_to_egovehicle_se3, map_range)
    if is_add_target_lanes:
        ax = add_target_lanes(ax, idx, log_id, map_range, city_name)
    if is_add_all_lanes:
        ax = add_lanes(xcenter, ycenter, r, city_name)
    if is_add_self:
        ax = add_self(ax, xcenter, ycenter)
    if save:
        save_map(fig, ax, idx=idx, log_index=log_index, save_jpg_npy=save_jpg_npy, dir=dir)
    else:
        fig.show()


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def get_prediction_grid():
    predictions_grid = np.zeros((70, 5, 5))
    predictions_grid[:, 2, 4] = 0.5
    predictions_grid[:, 1, 3] = 0.5
    for i in range(0, predictions_grid.shape[0]):
        if i >= 35:
            predictions_grid[i, 2, 4] *= min(max(1 + (i / 50), 0), 1)
            predictions_grid[i, 1, 3] *= min(max(1 - (i / 50), 0), 1)
            # print(predictions_grid[i, 2, 4], predictions_grid[i, 1, 3])
    return predictions_grid


def visualize_predictions_grid(vis_predictions_grid=True, vis_observed_single_point=True, vis_observed_sequence=False,
                               vis_predicted_traj=False, add_noise=False, save=False, path=None):
    ref_frame = 5
    city_to_egovehicle_se3 = argoverse_data.get_pose(ref_frame)
    xcenter, ycenter, _ = argoverse_data.get_pose(ref_frame).translation
    r = 25
    map_range = [xcenter - 5, xcenter + r + 20, ycenter - r, ycenter + r]  # [xmin, xmax, ymin, ymax]

    scene_input, social_input, person_input, expected_output = generate_network_input_data(preview=False, save_to_pickle=False)
    if path == 2:
        ex = np.zeros_like(expected_output)
        for idx in range(0, 25):
            ex[idx, :, 0] = np.linspace(expected_output[idx, 0, 0], 0.3, num=expected_output.shape[1])
            ex[idx, :, 1] = np.linspace(expected_output[idx, 0, 1], 0.64, num=expected_output.shape[1])
        for idx in range(25, 70):
            ex[idx, :, 0] = np.linspace(100, 100, num=expected_output.shape[1])
            ex[idx, :, 1] = np.linspace(100, 100, num=expected_output.shape[1])

        expected_output = ex

    predictions_grid = get_prediction_grid()

    img_dim = 496
    path = os.path.join(visualization_dir, "map_only", "{}_{}.jpg".format(log_index, ref_frame))
    img = Image.open(path)
    img = rgb2gray(np.flip(np.reshape(np.array(img.getdata()), (img_dim, img_dim, 3)), axis=0))

    for idx in range(0, 70, 1):
        if idx < 20:
            expected_output_frame = np.copy(expected_output[0])
            person_input_frame = np.copy(person_input[0])
        else:
            expected_output_frame = np.copy(expected_output[idx - 20])
            person_input_frame = np.copy(person_input[idx - 20])

        predictions_grid_frame = np.copy(predictions_grid[idx])
        predictions_grid_frame = predictions_grid_frame / np.sum(predictions_grid_frame)
        # expected_output_frame = expected_output_frame / np.sum(expected_output_frame)
        person_input_frame *= img_dim
        expected_output_frame *= img_dim
        fig, ax = get_plot([0, img_dim, 0, img_dim], axis_off=False, for_network=False)
        if vis_predictions_grid:
            ax.imshow(resize(predictions_grid_frame.T, (img_dim, img_dim), anti_aliasing=False, order=0), cmap='gray', alpha=1, vmin=0, vmax=1)  # OrRd
        ax.imshow(img, alpha=0.2)
        ax.set_title(idx)

        if vis_observed_sequence:
            ax.scatter(person_input_frame[:, 0], person_input_frame[:, 1], c="g")
        if vis_predicted_traj:
            if add_noise:
                noise1 = np.random.rand(*expected_output_frame[:, 0].shape) * 3
                noise2 = np.random.rand(*expected_output_frame[:, 0].shape) * 3
                ax.scatter(expected_output_frame[:, 0] + noise1, expected_output_frame[:, 1] + noise2, c="b")
            else:
                ax.scatter(expected_output_frame[:, 0], expected_output_frame[:, 1], c="b")

        if vis_observed_single_point:
            ax.scatter(person_input_frame[-1, 0], person_input_frame[-1, 1], c="r", s=100)

        xcenter, ycenter, _ = argoverse_data.get_pose(idx).translation
        xcenter, ycenter = normalize(xcenter, ycenter, map_range)
        xcenter, ycenter = xcenter * img_dim, ycenter * img_dim
        ax.scatter(xcenter, ycenter, c="b", s=150)

        if save:
            # save_map(fig, ax, idx=idx, log_index=log_index, save_jpg_npy="jpg", dir="predictions_trajectory1_noise")
            save_map(fig, ax, idx=idx, log_index=log_index, save_jpg_npy="jpg", dir="predictions_grid")
        else:
            fig.show()

visualize_predictions_grid(vis_predictions_grid=True, vis_observed_single_point=True, vis_observed_sequence=False,
                           vis_predicted_traj=False, add_noise=False, save=True, path=1)
# %%
def save_rgb_images_and_lanes(idx, log_index, log_id, camera, visualization_dir, generate_lanes=True, generate_bbox=False, save_lanes=False, save_bbox=False, final_centerlines=None, alpha_transparencies=None):
    img = argoverse_data.get_image_sync(idx, camera=camera)
    # Image.fromarray(img).save(os.path.join(visualization_dir, "{}_{}".format(camera, "rgb"), "{}_{}.jpg".format(log_index, idx)))
    # plt.imshow(img)
    # plt.title(idx)
    # plt.show()

    objects = argoverse_data.get_label_object(idx)
    if generate_lanes:
        city_to_egovehicle_se3 = argoverse_data.get_pose(idx)
        lidar_pts = argoverse_data.get_lidar(idx)

        if final_centerlines is not None:
            img_wlane = plot_lane_centerlines_in_img(lidar_pts, city_to_egovehicle_se3, img, city_name, argoverse_map, calib.camera_config, planes,
                                                     color=(255, 0, 0), linewidth=15, local_centerlines_list=final_centerlines, alpha_transparencies=alpha_transparencies)
        else:
            img_wlane = plot_lane_centerlines_in_img(lidar_pts, city_to_egovehicle_se3, img, city_name, argoverse_map, calib.camera_config, planes)

        if save_lanes:
            Image.fromarray(img_wlane).save(os.path.join(visualization_dir, "predictions_rgb_lanes", "{}_{}.jpg".format(log_index, idx)))
        else:
            plt.imshow(img_wlane)
            plt.show()
        # # plt.title(idx)
        # # plt.show()
        # # display(Image.fromarray(img_wlane))

    if generate_bbox:
        img_vis = viz_util.show_image_with_boxes(img, objects, calib)
        if save_bbox:
            Image.fromarray(img_vis).save(os.path.join(visualization_dir, "predictions_rgb_bbox", "{}_{}.jpg".format(log_index, idx)))
        else:
            display(Image.fromarray(img_vis))
    return


def generate_rgb_with_lanes():

    ref_frame = 5
    xcenter, ycenter, _ = argoverse_data.get_pose(ref_frame).translation
    r = 25
    map_range = [xcenter - 5, xcenter + r + 20, ycenter - r, ycenter + r]

    city_to_egovehicle_se3 = argoverse_data.get_pose(ref_frame)
    position, _ = get_target_vehicle_position(ref_frame, log_id, map_range, None, city_to_egovehicle_se3, is_normalized=False)
    [query_x, query_y] = position[0]
    local_centerlines = argoverse_map.find_local_lane_centerlines(query_x, query_y, city_name, query_search_range_manhattan=11)
    print(f"local_centerlines {len(local_centerlines)}")

    predictions_grid = get_prediction_grid()
    alpha_transparencies = np.zeros((70, 3))
    final_centerlines_per_frame = []
    for frame in range(0, 70):
        final_centerlines = []
        centerline = []
        indexes = [2, 6, 7, 8, 13, 14]
        if frame <= 8:
            indexes1 = [[2, 8], [6, 14], [7, 13]]
        elif frame <= 19:
            indexes1 = [[2, 8], [6, 14], [7, 13]]
        else:
            indexes1 = [[2, 8], [6, 14], []]

        for indexes in indexes1:
            for i, lane in enumerate(local_centerlines):
                if i in indexes:
                    centerline.append(lane)
            final_centerlines.append(centerline)
            centerline = []

        final_centerlines_per_frame.append(final_centerlines)

        alpha_transparencies[frame, 0] = predictions_grid[frame, 2, 4]
        alpha_transparencies[frame, 1] = predictions_grid[frame, 1, 3]
        alpha_transparencies[frame, 2] = 1.

    for idx in range(40, 70, 1):
        save_rgb_images_and_lanes(idx, log_index, log_id, camera, visualization_dir, generate_lanes=True, generate_bbox=False,
                                  save_lanes=False, save_bbox=True, final_centerlines=final_centerlines_per_frame[idx],
                                  alpha_transparencies=alpha_transparencies[idx])





# save_rgb_images_and_lanes(10, log_index, log_id, camera, visualization_dir, final_centerlines=[final_centerlines[2]])
# for idx in range(0,len(argoverse_data.lidar_timestamp_list), 5):
# for idx in range(0, 1, 1):
    # save_rgb_images_and_lanes(idx, log_index, log_id, camera, visualization_dir, generate_lanes=False, generate_bbox=True,
    #                           save_lanes=False, save_bbox=False)

def build_final_img_sequence():
    canvas_size = (1000, 1000, 3)
    idx = 30
    for idx in range(0,70):
        # img_bbox = os.path.join(visualization_dir, "predictions_rgb_bbox", "{}_{}.jpg".format(log_index, idx))
        # img_bbox = cv2.imread(img_bbox)
        img_lanes = os.path.join(visualization_dir, "predictions_rgb_lanes", "{}_{}.jpg".format(log_index, idx))
        img_lanes = mpimg.imread(img_lanes)
        img_grid = os.path.join(visualization_dir, "predictions_grid", "{}_{}.jpg".format(log_index, idx))
        img_grid = mpimg.imread(img_grid)
        img_traj1 = os.path.join(visualization_dir, "predictions_trajectory1_noise", "{}_{}.jpg".format(log_index, idx))
        img_traj1 = mpimg.imread(img_traj1)
        img_traj2 = os.path.join(visualization_dir, "predictions_trajectory2_noise", "{}_{}.jpg".format(log_index, idx))
        img_traj2 = mpimg.imread(img_traj2)

        # img_bbox = resize(img_bbox, (canvas_size[0]/2, canvas_size[0]/2), anti_aliasing=False, order=0)
        img_lanes = resize(img_lanes, (canvas_size[0]/2, canvas_size[0]/2), anti_aliasing=False, order=0)
        img_grid = resize(img_grid, (canvas_size[0]/2, canvas_size[0]/2), anti_aliasing=False, order=0)
        img_traj1 = resize(img_traj1, (canvas_size[0]/2, canvas_size[0]/2), anti_aliasing=False, order=0)
        img_traj2 = resize(img_traj2, (canvas_size[0]/2, canvas_size[0]/2), anti_aliasing=False, order=0)

        canvas = np.zeros(canvas_size)

        # img_grid
        x_offset, y_offset = 0, 0
        canvas[y_offset:y_offset + img_grid.shape[0], x_offset:x_offset + img_grid.shape[1]] = img_grid

        # img_lanes
        x_offset, y_offset = 0, int(canvas_size[0]/2)
        canvas[y_offset:y_offset + img_lanes.shape[0], x_offset:x_offset + img_lanes.shape[1]] = img_lanes

        # img_traj1
        x_offset, y_offset = int(canvas_size[0]/2), 0
        canvas[y_offset:y_offset + img_traj1.shape[0], x_offset:x_offset + img_traj1.shape[1]] = img_traj1

        # img_traj2
        x_offset, y_offset = int(canvas_size[0]/2), int(canvas_size[1]/2)
        canvas[y_offset:y_offset + img_traj2.shape[0], x_offset:x_offset + img_traj2.shape[1]] = img_traj2

        save_img_path = os.path.join(visualization_dir, "result_visualization", "{}.jpg".format(idx))

        # plt.imshow(canvas)

        my_dpi = 96.0  # screen constant, check here https://www.infobyip.com/detectmonitordpi.php
        fig = plt.figure(figsize=(canvas_size[0] / my_dpi, canvas_size[1] / my_dpi), dpi=my_dpi)

        # fig.tight_layout(pad=0)
        ax = fig.add_axes([0., 0., 1., 1.])
        ax.set_xlim([0, canvas_size[0]])
        ax.set_ylim([canvas_size[1], 0])
        ax.axis('off')
        ax.imshow(canvas)

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        img = Image.fromarray(data)
        img.save(save_img_path)


    # scipy.misc.imsave(save_img_path, canvas)
    return


build_final_img_sequence()
