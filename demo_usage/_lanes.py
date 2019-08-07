import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from cuboids_to_bboxes import plot_lane_centerlines_in_img
from PIL import Image
from argoverse.utils.frustum_clipping import generate_frustum_planes
import argoverse.visualization.visualization_utils as viz_util
import copy
import logging
from visualize_30hz_benchmark_data_on_map import DatasetOnMapVisualizer
from visualize_30hz_benchmark_data_on_map import draw_lane_polygons
from shapely.geometry.polygon import Polygon
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import numpy as np
from argoverse.utils.centerline_utils import (
    centerline_to_polygon,
    filter_candidate_centerlines,
    get_centerlines_most_aligned_with_trajectory,
    lane_waypt_to_query_dist,
    remove_overlapping_lane_seq,
)
from argoverse.utils.mpl_plotting_utils import plot_lane_segment_patch, visualize_centerline
from typing import Iterable, List, Sequence, Set, Tuple
from matplotlib.patches import Polygon
import os
import pickle


logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
# %%
am = ArgoverseMap()
tracking_dataset_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-tracking/sample/'
argoverse_loader = ArgoverseTrackingLoader(tracking_dataset_dir)
log_index = 0
log_id = argoverse_loader.log_list[log_index]
argoverse_data = argoverse_loader[log_index]
city_name = argoverse_data.city_name
# %%


class ArtificialDatasetGenerator:
    def __init__(self, argoverse_map=None, argoverse_loader=None):
        self.argoverse_map = argoverse_map
        self.argoverse_loader = argoverse_loader
        self.img_dataset_dir = "/media/bartosz/hdd1TB/workspace_hdd/SS-LSTM/data/argoverse/imgs_artificial"
        self.obs_frames, self.pred_frames = 20, 20

    def _generate_obs_pred(self):
        # for log_id in range(len(argoverse_loader.log_list)):
        #     log_id = argoverse_loader.log_list[log_index]
        #     argoverse_data = argoverse_loader[log_index]
        #     city_name = argoverse_data.city_name
        for city_name in ["PIT", "MIA"]:
            self.generate_for_city(city_name)

    def get_trajectories_from_cl(self, candidate_cl):
        trajectories = []
        for line_nb, centerline_coords in enumerate(candidate_cl):
            line_coords = list(zip(*centerline_coords))
            trajectories.append(line_coords)
            # laneX, laneY = line_coords[0], line_coords[1]
            # plt.plot(laneX, laneY, "--", color="grey", alpha=1, linewidth=1, zorder=0)

        return trajectories

    def interpolate_missing_frames(self, positions):
        interpolated_positions = []
        idxs = positions[:, 0]
        pos_x = positions[:, 1]
        pos_y = positions[:, 2]

        desired_idxs = np.arange(idxs[0], idxs[-1] + 1, 1)

        new_pos_x = np.interp(desired_idxs, idxs, pos_x)
        new_pos_y = np.interp(desired_idxs, idxs, pos_y)

        interpolated_positions = np.column_stack((desired_idxs, new_pos_x, new_pos_y))
        return interpolated_positions

    def interpolate_segments_to_steps(self, lane):
        # 1 lane consists of many segments
        # take 2 segments for obs and 2 for pred
        divide_segment_by = 5
        laneX, laneY = np.array(lane[0]), np.array(lane[1])

        all_pos = np.zeros((2, 0))
        #  loop segment by segment
        for ((xs_curr, ys_curr), (xs_next, ys_next)) in zip(zip(laneX, laneY), zip(laneX[1:], laneY[1:])):
            # dist = np.sqrt((xs_next-xs_curr)**2 + (ys_next-ys_curr)**2)
            idxs = [0, divide_segment_by]
            pos_x, pos_y = [xs_curr, xs_next], [ys_curr, ys_next]
            desired_idxs = np.arange(0, divide_segment_by + 1, 1)
            new_pos_x = np.interp(desired_idxs, idxs, pos_x)
            new_pos_y = np.interp(desired_idxs, idxs, pos_y)
            all_pos = np.hstack((all_pos, np.vstack((new_pos_x, new_pos_y))))

        # print(all_pos.shape)
        return all_pos

    def get_random_sample(self, trajectory, nsamples=1):
        trajectory = trajectory.T
        trajectory_length = trajectory.shape[0]
        desired_path_length = self.obs_frames + self.pred_frames

        starting_idxs = np.random.randint(0, trajectory_length - desired_path_length + 1, nsamples)

        res_trajectories = np.zeros((nsamples, desired_path_length, 2))
        for i, starting_idx in enumerate(starting_idxs):
            end_idx = starting_idx + desired_path_length
            res_trajectories[i] = trajectory[starting_idx:end_idx]

        # print(res_trajectories.shape, res_trajectories[:, 0:self.obs_frames].shape, res_trajectories[:, self.obs_frames:].shape)
        return res_trajectories[:, 0:self.obs_frames], res_trajectories[:, self.obs_frames:]

    def split_to_trajectories(self, single_lane, offset=8):
        positions = single_lane.T
        # print("single_lane {}".format(positions.shape))
        path_length = self.obs_frames + self.pred_frames
        unique_trajectories = int(max(0, ((positions.shape[0] - path_length) / offset) + 1))
        # print("unique_trajectories {}".format(unique_trajectories))

        obs_pos, pred_pos, obs_pred_pos = None, None, None
        for i in range(unique_trajectories):
            start_idx = i * offset
            end_idx = start_idx + self.obs_frames
            obs_single = positions[start_idx:end_idx]
            obs_single = np.expand_dims(obs_single, axis=0)
            # circle_occupancy_map_single = self._get_circle_occupancy_map_single(target_object, start_idx)
            # circle_occupancy_map_single = np.expand_dims(circle_occupancy_map_single, axis=0)

            start_idx = end_idx
            end_idx = start_idx + self.pred_frames
            pred_single = positions[start_idx:end_idx]
            pred_single = np.expand_dims(pred_single, axis=0)

            obs_pos = np.vstack((obs_pos, obs_single)) if obs_pos is not None else obs_single
            pred_pos = np.vstack((pred_pos, pred_single)) if pred_pos is not None else pred_single
            # circle_occupancy_map = np.vstack((circle_occupancy_map, circle_occupancy_map_single)) if circle_occupancy_map is not None else circle_occupancy_map_single

        # obs_pred_pos = np.concatenate((obs_pos, pred_pos), axis=1)
        return obs_pos, pred_pos

    def get_lanes_before_after(self, lane_candidates, id=None):
        if id is not None:
            lane_candidates = [lane_candidates[id]]

        lanes_around = []
        for lane in lane_candidates:
            dfs_threshold = 1
            candidates_future = am.dfs(lane, city_name, 0, dfs_threshold)
            candidates_past = am.dfs(lane, city_name, 0, dfs_threshold, True)
            # Merge past and future
            for past_lane_seq in candidates_past:
                for future_lane_seq in candidates_future:
                    assert past_lane_seq[-1] == future_lane_seq[0], "Incorrect DFS for candidate lanes past and future"
                    lanes_around.append(past_lane_seq + future_lane_seq[1:])
        return lanes_around

    def _normalize_positions(self, positions, map_range, verbose=False):
        [xmin, xmax, ymin, ymax] = map_range
        # p_copy = np.copy(positions)
        positions[..., 0] = (positions[..., 0] - xmin) / (xmax - xmin)
        positions[..., 1] = (positions[..., 1] - ymin) / (ymax - ymin)

    # def plot(self, trajectories_obs, trajectories_pred, full_trajectory=None):
    #     for (obs_pos, pred_pos) in zip(trajectories_obs, trajectories_pred):
    #
    #         fig = plt.figure(figsize=(7, 7))
    #         ax = fig.add_subplot(111)
    #
    #         # take one point and display driveable areas around it
    #         # starting_point_idx = 3
    #         # xcenter, ycenter = laneX[starting_point_idx], laneY[starting_point_idx]
    #         xcenter, ycenter = obs_pos[0, 0], obs_pos[0, 1]
    #         r = 25
    #         xmin, xmax = xcenter - r, xcenter + r
    #         ymin, ymax = ycenter - r, ycenter + r
    #         ax.set_xlim([xmin, xmax])
    #         ax.set_ylim([ymin, ymax])
    #
    #         local_lane_polygons = am.find_local_lane_polygons([xmin, xmax, ymin, ymax], city_name)
    #         ax = self.add_map(ax, local_lane_polygons)
    #
    #         if full_trajectory is not None:
    #             # vizualize full trajectory
    #             ax.plot(full_trajectory[0], full_trajectory[1], "--", color="red", alpha=1, linewidth=1)
    #
    #         # vizualize points on trajectory
    #         ax.scatter(obs_pos[:, 0], obs_pos[:, 1], color="green", zorder=2)
    #         ax.scatter(pred_pos[:, 0], pred_pos[:, 1], color="blue", zorder=2)
    #
    #         plt.show()

    def add_map(self, ax, map_polygons):
        # ax.set_facecolor("black")
        for i in range(0, len(map_polygons)):
            poly = map_polygons[i]
            ax.add_patch(Polygon(poly[:, 0:2], facecolor="black", alpha=1))

        return ax

    def get_map_range(self, obs_pos):
        xcenter, ycenter = obs_pos[0, 0], obs_pos[0, 1]
        r = 25
        xmin, xmax = xcenter - r, xcenter + r
        ymin, ymax = ycenter - r, ycenter + r
        return [xmin, xmax, ymin, ymax]

    @staticmethod
    def rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def get_and_save_img(self, id, trajectory_obs, trajectory_pred, map_range, preview=False):
        uuid = str(id).zfill(6)
        [xmin, xmax, ymin, ymax] = map_range
        local_lane_polygons = am.find_local_lane_polygons(map_range, city_name)
        if preview:
            fig = plt.figure(figsize=(5, 5))
        else:
            fig = plt.figure(figsize=(1, 1))

        ax = fig.add_axes([0., 0., 1., 1.])
        ax.set_xlim([map_range[0], map_range[1]])
        ax.set_ylim([map_range[2], map_range[3]])
        ax = self.add_map(ax, local_lane_polygons)

        if preview:
            ax.scatter(trajectory_obs[:, 0] * (xmax - xmin) + xmin, trajectory_obs[:, 1] * (ymax - ymin) + ymin, zorder=20)
            ax.scatter(trajectory_pred[:, 0] * (xmax - xmin) + xmin, trajectory_pred[:, 1] * (ymax - ymin) + ymin, zorder=20)
            return ""
        else:
            ax.axis('off')

            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            data = 1 - self.rgb2gray(data)

            img_path = os.path.join(self.img_dataset_dir, 'scene_{}.npy'.format(uuid))

            np.save(img_path, data)
            plt.close()

            return img_path

    def generate_for_city(self, city_name):
        curr_lane_candidates = am.get_lane_ids_in_xy_bbox(0, 0, city_name, np.inf)
        print("all lanes {}".format(len(curr_lane_candidates)))

        lanes_around = self.get_lanes_before_after(curr_lane_candidates)
        candidate_cl = am.get_cl_from_lane_seq(lanes_around, city_name)
        full_trajectories = self.get_trajectories_from_cl(candidate_cl)

        # loop and stack
        person_input = np.zeros((0, 20, 2))
        expected_output = np.zeros((0, 20, 2))
        scene_input = []  # np.zeros((0, 20, 1))
        for lane_idx in range(len(full_trajectories)):
            if lane_idx > len(full_trajectories) + 1:
                break
            full_trajectory = full_trajectories[lane_idx]

            full_trajectory = self.interpolate_segments_to_steps(full_trajectory)
            # trajectories_obs, trajectories_pred = self.split_to_trajectories(full_trajectory)
            trajectories_obs, trajectories_pred = self.get_random_sample(full_trajectory)
            # self.plot(trajectories_obs, trajectories_pred, full_trajectory)

            for id, (trajectory_obs, trajectory_pred) in enumerate(zip(trajectories_obs, trajectories_pred)):
                map_range = self.get_map_range(trajectory_obs)
                self._normalize_positions(trajectory_obs, map_range)
                self._normalize_positions(trajectory_pred, map_range)

                img_path = self.get_and_save_img(id, trajectory_obs, trajectory_pred, map_range)

                scene_input.append(img_path)

            person_input = np.vstack((person_input, trajectories_obs))
            expected_output = np.vstack((expected_output, trajectories_pred))
            social_input = np.zeros((expected_output.shape[0], self.obs_frames, 64))

            # save checkpoint
            save_every = 100
            if (lane_idx + 1) % save_every == 0:
                # path = "/media/bartosz/hdd1TB/workspace_hdd/argoverse-api/demo_usage/artificial_data_checkpoints"
                # path = os.path.join(path, "data_checkpoint_{}.pickle".format(str(int(lane_idx / save_every)).zfill(6)))
                # self.save_to_pickle(path, [scene_input, social_input, person_input, expected_output])
                print("Progress: {}%".format(int(lane_idx / len(full_trajectories) * 100)))

        path = "/media/bartosz/hdd1TB/workspace_hdd/argoverse-api/demo_usage/artificial_data_checkpoints"
        path = os.path.join(path, "data_final_{}.pickle".format(city_name))
        self.save_to_pickle(path, [scene_input, social_input, person_input, expected_output])

        print(person_input.shape, expected_output.shape, len(scene_input))
        # person_input, expected_output, social_input, scene_input
        return scene_input, social_input, person_input, expected_output

    def save_to_pickle(self, name, array):
        # array in a form [scene_input, social_input, person_input, expected_output]
        pickle_out = open(name, "wb")
        pickle.dump(array, pickle_out, protocol=2)
        pickle_out.close()
        print("Saved to pickle {}".format(name))


if __name__ == "__main__":
    pass
    adg = ArtificialDatasetGenerator(argoverse_map=am, argoverse_loader=argoverse_loader)
    city_name = "MIA"
    scene_input, social_input, person_input, expected_output = adg.generate_for_city(city_name)

    # merge_lists()


def merge_lists():
    # [scene_input, social_input, person_input, expected_output]
    city_name = ["PIT", "MIA"]
    path = "/media/bartosz/hdd1TB/workspace_hdd/argoverse-api/demo_usage/artificial_data_checkpoints"
    path1 = os.path.join(path, "data_final_{}.pickle".format(city_name[0]))
    path2 = os.path.join(path, "data_final_{}.pickle".format(city_name[1]))

    pickle_in = open(path1, "rb")
    data1 = pickle.load(pickle_in)
    print(data1[1].shape)

    pickle_in = open(path2, "rb")
    data2 = pickle.load(pickle_in)
    print(data2[1].shape)

    scene_input = data1[0] + data2[0]
    social_input = np.vstack((data1[1], data2[1]))
    person_input = np.vstack((data1[2], data2[2]))
    expected_output = np.vstack((data1[3], data2[3]))

    print(len(scene_input), social_input.shape, person_input.shape, expected_output.shape)

    path_res = os.path.join(path, "data_final_{}_{}.pickle".format(city_name[0], city_name[1]))

    pickle_out = open(path_res, "wb")
    pickle.dump([scene_input, social_input, person_input, expected_output], pickle_out, protocol=2)
    pickle_out.close()
    print("Saved to pickle {}".format(path_res))
