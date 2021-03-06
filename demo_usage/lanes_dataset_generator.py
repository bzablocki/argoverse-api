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
import sys
import pickle
import random


logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


# %%
class ArtificialDatasetGenerator:
    def __init__(self, argoverse_map=None, argoverse_loader=None, grid_map_input=False, dev_mode=False):
        self.argoverse_map = argoverse_map
        self.argoverse_loader = argoverse_loader
        self.img_dataset_dir = "/media/bartosz/hdd1TB/workspace_hdd/SS-LSTM/data/argoverse/imgs_artificial_v6_2030"
        self.cache_dir = "/media/bartosz/hdd1TB/workspace_hdd/argoverse-api/demo_usage/artificial_data_checkpoints_v6"
        self.obs_frames, self.pred_frames = 20, 30
        self.dev_mode = dev_mode
        self.grid_map_input = grid_map_input

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

    # def interpolate_missing_frames(self, positions):
    #     interpolated_positions = []
    #     idxs = positions[:, 0]
    #     pos_x = positions[:, 1]
    #     pos_y = positions[:, 2]
    #
    #     desired_idxs = np.arange(idxs[0], idxs[-1] + 1, 1)
    #
    #     new_pos_x = np.interp(desired_idxs, idxs, pos_x)
    #     new_pos_y = np.interp(desired_idxs, idxs, pos_y)
    #
    #     interpolated_positions = np.column_stack((desired_idxs, new_pos_x, new_pos_y))
    #     return interpolated_positions

    def interpolate_segments_to_steps(self, lane):
        divide_segment_by = 1
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

        return all_pos

    def get_random_sample(self, trajectory, city_name, nsamples=1, validation_dataset=False):  # it's actually splitting trajectories
        trajectory = trajectory.T
        trajectory_length = trajectory.shape[0]
        desired_path_length = self.obs_frames + self.pred_frames
        if city_name == "MIA":
            offset = 18
        else:
            offset = 9
        unique_trajectories = np.floor(max(0, ((trajectory.shape[0] - desired_path_length) / offset) + 1)).astype(np.int)
        obs_pos = None
        pred_pos = None
        for i in range(unique_trajectories):
            start_idx = i * offset
            end_idx = start_idx + self.obs_frames
            obs_single = trajectory[start_idx:end_idx]
            obs_single = np.expand_dims(obs_single, axis=0)

            start_idx = end_idx
            end_idx = start_idx + self.pred_frames
            pred_single = trajectory[start_idx:end_idx]
            pred_single = np.expand_dims(pred_single, axis=0)

            obs_pos = np.vstack((obs_pos, obs_single)) if obs_pos is not None else obs_single
            pred_pos = np.vstack((pred_pos, pred_single)) if pred_pos is not None else pred_single

        if obs_pos is None or pred_pos is None:
            return [], []
        else:
            return obs_pos, pred_pos
        # if validation_dataset:
        #     np.random.seed(42)
        # else:
        #     np.random.seed(0)
        # starting_idxs = np.random.randint(0, trajectory_length - desired_path_length + 1, nsamples)
        # starting_idxs = np.unique(starting_idxs)
        #
        # res_trajectories = np.zeros((nsamples, desired_path_length, 2))
        # for i, starting_idx in enumerate(starting_idxs):
        #     end_idx = starting_idx + desired_path_length
        #     res_trajectories[i] = trajectory[starting_idx:end_idx]
        #
        # # print(res_trajectories.shape, res_trajectories[:, 0:self.obs_frames].shape, res_trajectories[:, self.obs_frames:].shape)
        # return res_trajectories[:, 0:self.obs_frames], res_trajectories[:, self.obs_frames:]

    # def split_to_trajectories(self, single_lane, offset=8):
    #     positions = single_lane.T
    #     # print("single_lane {}".format(positions.shape))
    #     path_length = self.obs_frames + self.pred_frames
    #     unique_trajectories = int(max(0, ((positions.shape[0] - path_length) / offset) + 1))
    #     # print("unique_trajectories {}".format(unique_trajectories))
    #
    #     obs_pos, pred_pos, obs_pred_pos = None, None, None
    #     for i in range(unique_trajectories):
    #         start_idx = i * offset
    #         end_idx = start_idx + self.obs_frames
    #         obs_single = positions[start_idx:end_idx]
    #         obs_single = np.expand_dims(obs_single, axis=0)
    #         # circle_occupancy_map_single = self._get_circle_occupancy_map_single(target_object, start_idx)
    #         # circle_occupancy_map_single = np.expand_dims(circle_occupancy_map_single, axis=0)
    #
    #         start_idx = end_idx
    #         end_idx = start_idx + self.pred_frames
    #         pred_single = positions[start_idx:end_idx]
    #         pred_single = np.expand_dims(pred_single, axis=0)
    #
    #         obs_pos = np.vstack((obs_pos, obs_single)) if obs_pos is not None else obs_single
    #         pred_pos = np.vstack((pred_pos, pred_single)) if pred_pos is not None else pred_single
    #         # circle_occupancy_map = np.vstack((circle_occupancy_map, circle_occupancy_map_single)) if circle_occupancy_map is not None else circle_occupancy_map_single
    #
    #     # obs_pred_pos = np.concatenate((obs_pos, pred_pos), axis=1)
    #     return obs_pos, pred_pos

    def get_lanes_before_after(self, lane_candidates, id=None):
        if id is not None:
            lane_candidates = [lane_candidates[id]]

        lanes_around = []
        for lane in lane_candidates:
            dfs_threshold = 100
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

    def plot(self, trajectories_obs, trajectories_pred, full_trajectory=None, only_full=False):

        for (obs_pos, pred_pos) in zip(trajectories_obs, trajectories_pred):

            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111)

            # take one point and display driveable areas around it
            # starting_point_idx = 3
            # xcenter, ycenter = laneX[starting_point_idx], laneY[starting_point_idx]
            xcenter, ycenter = obs_pos[0, 0], obs_pos[0, 1]
            r = 25 if not self.dev_mode else 100
            xmin, xmax = xcenter - r, xcenter + r
            ymin, ymax = ycenter - r, ycenter + r
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])

            local_lane_polygons = am.find_local_lane_polygons([xmin, xmax, ymin, ymax], city_name)
            ax = self.add_map(ax, local_lane_polygons)

            if full_trajectory is not None:
                # vizualize full trajectory
                ax.plot(full_trajectory[0], full_trajectory[1], "--", color="red", alpha=1, linewidth=1)
            if only_full:
                plt.show()
                break

            # vizualize points on trajectory
            ax.scatter(obs_pos[:, 0], obs_pos[:, 1], color="green", zorder=2)
            ax.scatter(pred_pos[:, 0], pred_pos[:, 1], color="blue", zorder=2)

            plt.show()

    def add_map(self, ax, map_polygons):
        # ax.set_facecolor("black")
        for i in range(0, len(map_polygons)):
            poly = map_polygons[i]
            ax.add_patch(Polygon(poly[:, 0:2], facecolor="black", alpha=1))

        return ax

    def get_map_range(self, obs_pos):
        xcenter, ycenter = obs_pos[-1, 0], obs_pos[-1, 1]
        r = 40.
        xmin, xmax = xcenter - r, xcenter + r
        ymin, ymax = ycenter - r, ycenter + r
        return [xmin, xmax, ymin, ymax]

    @staticmethod
    def rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def get_and_save_img(self, lane_idx, traj_nb, city_name, trajectory_obs, trajectory_pred, map_range, direction="",
                         preview=False, save=False, force_save=False, validation_dataset=False, img_title=""):
        uuid = city_name + "_" + str(lane_idx).zfill(6) + "_" + str(traj_nb).zfill(2)
        if validation_dataset:
            img_path = os.path.join(self.img_dataset_dir, 'scene_val_{}.npy'.format(uuid))
        else:
            img_path = os.path.join(self.img_dataset_dir, 'scene_{}.npy'.format(uuid))

        if not (force_save or not os.path.exists(img_path) or preview):
            return img_path

        [xmin, xmax, ymin, ymax] = map_range
        local_lane_polygons = am.find_local_lane_polygons(map_range, city_name)
        if preview:
            fig = plt.figure(figsize=(5, 5))
        else:
            # fig = plt.figure(figsize=(1, 1))
            my_dpi = 96.0
            fig = plt.figure(figsize=(72 / my_dpi, 72 / my_dpi), dpi=my_dpi)

        ax = fig.add_axes([0., 0., 1., 1.])
        ax.set_xlim([map_range[0], map_range[1]])
        ax.set_ylim([map_range[2], map_range[3]])
        ax = self.add_map(ax, local_lane_polygons)

        if preview:
            ax.set_title(img_title)
            ax.scatter(trajectory_obs[:, 0] * (xmax - xmin) + xmin, trajectory_obs[:, 1] * (ymax - ymin) + ymin, zorder=20, c='g')
            ax.scatter(trajectory_pred[:, 0] * (xmax - xmin) + xmin, trajectory_pred[:, 1] * (ymax - ymin) + ymin, zorder=20, c='b')
            return ""
        else:
            ax.axis('off')

            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            data = np.rot90(data.transpose(1, 0, 2))
            data = 1 - self.rgb2gray(data)

            if data.shape[0] != 72 and data.shape[1] != 72:
                print("Wrong data shape {} {}".format(data.shape, img_path))

            if save:
                if not os.path.exists(img_path) or force_save:
                    # print("save {}".format(img_path))
                    np.save(img_path, data)

            plt.close()
            # print(data.shape, img_path)
            return img_path

    def is_within_range(self, trajectory_obs, trajectory_pred):
        if np.any(trajectory_obs <= 0) or np.any(trajectory_obs >= 1) or np.any(trajectory_pred <= 0) or np.any(trajectory_pred >= 1):
            return False
        return True

    @staticmethod
    def normalize_vector(vec):
        return vec / np.linalg.norm(vec)

    @staticmethod
    def string_direction(perp_input_vector, input_vector, final_vector):
        perp_input_dot_final = np.dot(perp_input_vector, final_vector)
        mean_input_dot_final = np.dot(input_vector, final_vector)

        # bins = np.linspace(-1, 1, 6, endpoint=True) + [0, 0, -0.07, 0.07, 0, 0]
        bins = np.linspace(-1, 1, 4, endpoint=True)
        turn_bin_nb = np.where(bins < perp_input_dot_final)[0][-1]

        is_turning_around = mean_input_dot_final < 0
        if is_turning_around:
            # labels = ["return_hard_left", "return_light_left", "return_straigth", "return_light_right", "return_hard_right"]
            labels = ["right", "return_straight", "left"]
        else:
            # labels = ["hard_left", "light_left", "straigth", "light_right", "hard_right"]
            labels = ["right", "straight", "left"]

        return labels[turn_bin_nb], perp_input_dot_final

    def get_direction(self, trajectory_obs, trajectory_pred):
        # print(trajectory_obs.shape) (20,2)
        input_vector = self.normalize_vector(trajectory_obs[-1] - trajectory_obs[0])
        final_vector = self.normalize_vector(trajectory_pred[-1] - trajectory_pred[0])
        perp_input_vector = [-input_vector[1], input_vector[0]]
        string_direction, dot_value = self.string_direction(perp_input_vector, input_vector, final_vector)
        return string_direction

    def get_subset_to_save(self, person_input_by_direction, expected_output_by_direction, scene_input_by_direction):
        print("\nBefore")
        for direction in person_input_by_direction:
            person_input_items = person_input_by_direction[direction]
            expected_output_by_direction_items = expected_output_by_direction[direction]
            scene_input_by_direction_items = scene_input_by_direction[direction]
            print("\t{} -> {} | {} | {}".format(direction, person_input_items.shape, expected_output_by_direction_items.shape, scene_input_by_direction_items.shape))

        person_input_to_save, expected_output_to_save, scene_input_to_save = np.zeros((0, self.obs_frames, 2)), np.zeros((0, self.pred_frames, 2)), np.zeros((0, 1))
        shortest_key = ""
        shortest_length = np.inf
        for direction in person_input_by_direction:
            items = person_input_by_direction[direction]
            if items.shape[0] < shortest_length:
                shortest_length = items.shape[0]
                shortest_key = direction

        print("\nShortest: {}, {} elements".format(shortest_key, shortest_length))

        for direction in person_input_by_direction:
            person_input_items = person_input_by_direction[direction]
            expected_output_items = expected_output_by_direction[direction]
            scene_input_items = scene_input_by_direction[direction]

            indexes_to_save = random.sample(range(person_input_items.shape[0]), shortest_length)
            # print("{} -> {}".format(direction, indexes_to_save))
            person_input_to_save = np.vstack((person_input_to_save, person_input_items[indexes_to_save]))
            expected_output_to_save = np.vstack((expected_output_to_save, expected_output_items[indexes_to_save]))
            scene_input_to_save = np.vstack((scene_input_to_save, scene_input_items[indexes_to_save]))

        # shuffle
        indexes_shuffled = random.sample(range(scene_input_to_save.shape[0]), scene_input_to_save.shape[0])
        person_input_to_save = person_input_to_save[indexes_shuffled]
        expected_output_to_save = expected_output_to_save[indexes_shuffled]
        scene_input_to_save = scene_input_to_save[indexes_shuffled]

        print("\nAfter")
        print("\t{} | {} | {}".format(person_input_to_save.shape, expected_output_to_save.shape, scene_input_to_save.shape))
        print("")
        return person_input_to_save, expected_output_to_save, scene_input_to_save

    def save_to_pickle(self, name, array, verbose=True):
        # array in a form [scene_input, social_input, person_input, expected_output]
        pickle_out = open(name, "wb")
        pickle.dump(array, pickle_out, protocol=2)
        pickle_out.close()
        if verbose:
            print("Saved to pickle {}".format(name))

    def get_new_io_dicts(self):
        person_input_by_direction = {"left": np.zeros((0, self.obs_frames, 2)), "straight": np.zeros((0, self.obs_frames, 2)), "right": np.zeros((0, self.obs_frames, 2))}
        expected_output_by_direction = {"left": np.zeros((0, self.pred_frames, 2)), "straight": np.zeros((0, self.pred_frames, 2)), "right": np.zeros((0, self.pred_frames, 2))}
        scene_input_by_direction = {"left": np.zeros((0, 1)), "straight": np.zeros((0, 1)), "right": np.zeros((0, 1))}
        return person_input_by_direction, expected_output_by_direction, scene_input_by_direction

    def merge_cache(self, city, curr_lane_candidates=None, is_validation=False, save=True):
        if is_validation:
            random.seed(10)
        else:
            random.seed(42)

        merge_every = 1500

        if is_validation:
            take_n_percentage = 0.005 if city == "MIA" else 0.005
        else:
            take_n_percentage = 0.3 if city == "MIA" else 0.4


        if curr_lane_candidates is None:
            curr_lane_candidates = am.get_lane_ids_in_xy_bbox(0, 0, city_name, np.inf)
        print("generating all the lanes: {} lanes in total".format(len(curr_lane_candidates)))

        lanes_around = self.get_lanes_before_after(curr_lane_candidates)
        candidate_cl = am.get_cl_from_lane_seq(lanes_around, city_name)
        full_trajectories = self.get_trajectories_from_cl(candidate_cl)
        print("full_trajectories {}".format(len(full_trajectories)))

        # scenes_omitted = 0
        person_input_by_direction_final, expected_output_by_direction_final, scene_input_by_direction_final = self.get_new_io_dicts()
        person_input_tmp, expected_output_tmp, scene_input_tmp = self.get_new_io_dicts()

        for lane_idx in range(len(full_trajectories)):
        # for lane_idx in range(20):

            cache_lane_path = os.path.join(self.cache_dir, "{}_{}".format(city_name, lane_idx))
            if os.path.exists(cache_lane_path):
                pickle_in = open(cache_lane_path, "rb")
                pickle_data = pickle.load(pickle_in)
                [person_input_pickle, expected_output_pickle, scene_input_pickle] = pickle_data
                for direction in ["left", "straight", "right"]:
                    # n = int(person_input_pickle[direction].shape[0])
                    # idxs = random.sample(range(n), int(n * take_n_percentage))
                    person_input_tmp[direction] = np.vstack((person_input_tmp[direction], person_input_pickle[direction]))
                    expected_output_tmp[direction] = np.vstack((expected_output_tmp[direction], expected_output_pickle[direction]))
                    scene_input_tmp[direction] = np.vstack((scene_input_tmp[direction], scene_input_pickle[direction]))

            if (lane_idx + 1) % merge_every == 0 or lane_idx == len(full_trajectories)-1:
                for direction in ["left", "straight", "right"]:
                    n = int(person_input_tmp[direction].shape[0])
                    idxs = random.sample(range(n), int(n * take_n_percentage))
                    # print(f"Before {direction}: {person_input_tmp[direction].shape[0]} |\t After: {person_input_tmp[direction][idxs].shape[0]}")
                    person_input_by_direction_final[direction] = np.vstack((person_input_by_direction_final[direction], person_input_tmp[direction][idxs]))
                    expected_output_by_direction_final[direction] = np.vstack((expected_output_by_direction_final[direction], expected_output_tmp[direction][idxs]))
                    scene_input_by_direction_final[direction] = np.vstack((scene_input_by_direction_final[direction], scene_input_tmp[direction][idxs]))
                print("")
                person_input_tmp, expected_output_tmp, scene_input_tmp = self.get_new_io_dicts()

            log_every = 1500
            if (lane_idx + 1) % log_every == 0:
                print("Progress: {}% ".format(int(lane_idx / len(full_trajectories) * 100)))

                for direction in ["left", "straight", "right"]:
                    print(f"person_input[{direction}]:{person_input_by_direction_final[direction].shape} | expected_output:{expected_output_by_direction_final[direction].shape} | scene_input:{scene_input_by_direction_final[direction].shape} |")
                print("")

        person_input, expected_output, scene_input = self.get_subset_to_save(person_input_by_direction_final, expected_output_by_direction_final, scene_input_by_direction_final)
        social_input = np.zeros((person_input.shape[0], self.obs_frames, 64))

        path = "/media/bartosz/hdd1TB/workspace_hdd/argoverse-api/demo_usage/artificial_data_checkpoints_v6"
        prefix = "data_final_v6_2030"
        if is_validation:
            path = os.path.join(path, "{}_val_{}.pickle".format(prefix, city))
        else:
            path = os.path.join(path, "{}_{}.pickle".format(prefix, city))

        if save:
            self.save_to_pickle(path, [scene_input, social_input, person_input, expected_output])

        return

    def generate_for_city(self, city_name, save=False, preview=False, validation_dataset=False, curr_lane_candidates=None, use_cache=False, is_reversed=False, start_from=None):
        if curr_lane_candidates is None:
            curr_lane_candidates = am.get_lane_ids_in_xy_bbox(0, 0, city_name, np.inf)
            print("generating all the lanes: {} lanes in total".format(len(curr_lane_candidates)))

        lanes_around = self.get_lanes_before_after(curr_lane_candidates)
        candidate_cl = am.get_cl_from_lane_seq(lanes_around, city_name)
        full_trajectories = self.get_trajectories_from_cl(candidate_cl)
        # loop and stack
        person_input = np.zeros((0, self.obs_frames, 2))
        expected_output = np.zeros((0, self.pred_frames, 2))
        scene_input = []
        straight_counter = 0

        person_input_by_direction = {"left": np.zeros((0, self.obs_frames, 2)), "straight": np.zeros((0, self.obs_frames, 2)), "right": np.zeros((0, self.obs_frames, 2))}
        expected_output_by_direction = {"left": np.zeros((0, self.pred_frames, 2)), "straight": np.zeros((0, self.pred_frames, 2)), "right": np.zeros((0, self.pred_frames, 2))}
        scene_input_by_direction = {"left": np.zeros((0, 1)), "straight": np.zeros((0, 1)), "right": np.zeros((0, 1))}

        if is_reversed:
            range_loop = range(len(full_trajectories) - 1, -1, -1)
        else:
            if start_from is None:
                range_loop = range(len(full_trajectories))
            else:
                range_loop = range(start_from, len(full_trajectories))

        # total_avg = []
        for lane_idx in range_loop:
            has_cache = False
            cache_lane_path = os.path.join(self.cache_dir, "{}_{}".format(city_name, lane_idx))
            if not self.dev_mode and use_cache:
                if os.path.exists(cache_lane_path):
                    has_cache = True

            if use_cache and has_cache:
                print("Has cache for {}, skipping.".format(lane_idx))
                continue

            if self.dev_mode and lane_idx > 20:
                break

            # print("lane_idx {}".format(lane_idx))
            full_trajectory = full_trajectories[lane_idx]

            full_trajectory = self.interpolate_segments_to_steps(full_trajectory)
            trajectories_obs, trajectories_pred = self.get_random_sample(full_trajectory, city_name, nsamples=5, validation_dataset=validation_dataset)
            # print("trajectories_obs.shape {}".format(trajectories_obs.shape))
            if self.dev_mode:
                self.plot(trajectories_obs, trajectories_pred, full_trajectory, only_full=True)

            # distances = []
            for traj_nb, (trajectory_obs, trajectory_pred) in enumerate(zip(trajectories_obs, trajectories_pred)):
                if not np.all(trajectory_obs[0] == trajectory_obs[-1]):

                    # data = np.concatenate((trajectory_obs, trajectory_pred), axis=0)
                    # for i in range(data.shape[0] - 1):
                    #     distances.append(np.linalg.norm(data[i] - data[i + 1], axis=0))
                    # total_avg.append(np.mean(distances))

                    map_range = self.get_map_range(trajectory_obs)
                    self._normalize_positions(trajectory_obs, map_range)
                    self._normalize_positions(trajectory_pred, map_range)
                    direction = self.get_direction(trajectory_obs, trajectory_pred)
                    # print(direction)
                    if direction == "straight":
                        straight_counter += 1

                    # save only every third straight path
                    if self.dev_mode or ((direction == "straight" and straight_counter % 8 == 0) or direction != "straight"):
                        if self.is_within_range(trajectory_obs, trajectory_pred) and direction in person_input_by_direction:
                            img_path = self.get_and_save_img(lane_idx, traj_nb, city_name, trajectory_obs, trajectory_pred, map_range, direction=direction,
                                                             force_save=False, save=save, preview=preview if self.dev_mode else False, validation_dataset=validation_dataset)
                            img_path = np.array(img_path).astype('object')

                            person_input_by_direction[direction] = np.vstack((person_input_by_direction[direction], np.expand_dims(trajectory_obs, axis=0)))
                            expected_output_by_direction[direction] = np.vstack((expected_output_by_direction[direction], np.expand_dims(trajectory_pred, axis=0)))
                            scene_input_by_direction[direction] = np.vstack((scene_input_by_direction[direction], np.expand_dims(img_path, axis=0)))

            # print(f"Final avg {np.sum(total_avg)/len(total_avg)}, std: {np.std(total_avg)}, max: {np.max(total_avg)}, entries: {len(total_avg)}")

            # log checkpoint
            log_every = 100
            if (lane_idx + 1) % log_every == 0:
                text = "Progress: {}% // ".format(int(lane_idx / len(full_trajectories) * 100))
                # print("Progress: {}% // {}".format(int(lane_idx / len(full_trajectories) * 100), (person_input.shape, expected_output.shape, len(scene_input))))

                text += "Person input:"
                for direction in person_input_by_direction:
                    items = person_input_by_direction[direction]
                    text += " {} {} |".format(direction, items.shape)
                print(text)

            # save lane data
            if use_cache:
                self.save_to_pickle(cache_lane_path, [person_input_by_direction, expected_output_by_direction, scene_input_by_direction], verbose=True)
                person_input_by_direction = {"left": np.zeros((0, self.obs_frames, 2)), "straight": np.zeros((0, self.obs_frames, 2)), "right": np.zeros((0, self.obs_frames, 2))}
                expected_output_by_direction = {"left": np.zeros((0, self.pred_frames, 2)), "straight": np.zeros((0, self.pred_frames, 2)), "right": np.zeros((0, self.pred_frames, 2))}
                scene_input_by_direction = {"left": np.zeros((0, 1)), "straight": np.zeros((0, 1)), "right": np.zeros((0, 1))}


def merge_lists(is_validation=False):

    # [scene_input, social_input, person_input, expected_output]
    city_name = ["PIT", "MIA"]
    path = "/media/bartosz/hdd1TB/workspace_hdd/argoverse-api/demo_usage/artificial_data_checkpoints_v6"
    prefix = "data_final_v6_2030"
    if is_validation:
        path1 = os.path.join(path, "{}_val_{}.pickle".format(prefix, city_name[0]))
        path2 = os.path.join(path, "{}_val_{}.pickle".format(prefix, city_name[1]))
    else:
        path1 = os.path.join(path, "{}_{}.pickle".format(prefix, city_name[0]))
        path2 = os.path.join(path, "{}_{}.pickle".format(prefix, city_name[1]))

    pickle_in = open(path1, "rb")
    data1 = pickle.load(pickle_in)
    print(data1[0].shape)

    pickle_in = open(path2, "rb")
    data2 = pickle.load(pickle_in)
    print(data2[0].shape)

    # scene_input = data1[0] + data2[0]
    # scene_input = np.array(scene_input, dtype='object')
    scene_input = np.vstack((np.expand_dims(data1[0], axis=-1), np.expand_dims(data2[0], axis=-1)))
    social_input = np.vstack((data1[1], data2[1]))
    person_input = np.vstack((data1[2], data2[2]))
    expected_output = np.vstack((data1[3], data2[3]))

    print(scene_input.shape, social_input.shape, person_input.shape, expected_output.shape)
    if is_validation:
        path_res = os.path.join(path, "{}_val_{}_{}.pickle".format(prefix, city_name[0], city_name[1]))
    else:
        path_res = os.path.join(path, "{}_{}_{}.pickle".format(prefix, city_name[0], city_name[1]))

    pickle_out = open(path_res, "wb")
    pickle.dump([scene_input, social_input, person_input, expected_output], pickle_out, protocol=2)
    pickle_out.close()
    print("Saved to pickle {}".format(path_res))


if __name__ == "__main__":

    am = ArgoverseMap()
    tracking_dataset_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-tracking/sample/'
    argoverse_loader = ArgoverseTrackingLoader(tracking_dataset_dir)
    # log_index = 0
    # log_id = argoverse_loader.log_list[log_index]
    # argoverse_data = argoverse_loader[log_index]
    # city_name = argoverse_data.city_name
    # curr_lane_candidates = am.get_lane_ids_in_xy_bbox(0, 0, city_name, np.inf)

    city_name = "MIA"
    # adg = ArtificialDatasetGenerator(argoverse_map=am, argoverse_loader=argoverse_loader, dev_mode=False)
    # adg = ArtificialDatasetGenerator(argoverse_map=am, argoverse_loader=argoverse_loader, dev_mode=True)
    # adg.generate_for_city(city_name, save=False, preview=False, curr_lane_candidates=curr_lane_candidates)
    # adg.generate_for_city(city_name, save=True, curr_lane_candidates=None)
    # adg.merge_cache(city_name, is_validation=False, save=False)

    # %
    ########################
    city_names = ["PIT", "MIA"]
    if len(sys.argv) >= 2 and sys.argv[1] in city_names:
        city_name = sys.argv[1]
        is_reversed = False
        start_from = None
        if len(sys.argv) >= 3 and sys.argv[2] in ["True", "False"]:
            is_reversed = True if (len(sys.argv) >= 3 and sys.argv[2] == "True") else False
        if len(sys.argv) >= 3 and sys.argv[2].isdigit():
            start_from = int(sys.argv[2]) if sys.argv[2].isdigit() else None
        print("Processing for {} - is_reversed: {}".format(city_name, is_reversed))
        adg = ArtificialDatasetGenerator(argoverse_map=am, argoverse_loader=argoverse_loader, grid_map_input=True)
        adg.generate_for_city(city_name, save=True, use_cache=True, is_reversed=is_reversed, start_from=start_from)
    elif len(sys.argv) >= 3 and sys.argv[1] == "merge_cache" and sys.argv[2] in city_names:
        city_name = sys.argv[2]
        adg = ArtificialDatasetGenerator(argoverse_map=am, argoverse_loader=argoverse_loader, dev_mode=False)
        adg.merge_cache(city_name, is_validation=True, save=True)
    elif len(sys.argv) >= 2 and sys.argv[1] == "merge_lists":
        merge_lists(is_validation=True)
    else:
        # adg = ArtificialDatasetGenerator(argoverse_map=am, argoverse_loader=argoverse_loader)
        city_name = "MIA"
        # adg = ArtificialDatasetGenerator(argoverse_map=am, argoverse_loader=argoverse_loader, dev_mode=False)
        # adg.merge_cache(city_name, is_validation=False, save=False)
        # adg.generate_for_city(city_name, save=False, validation_dataset=False)

        # x = np.load("/media/bartosz/hdd1TB/workspace_hdd/SS-LSTM/data/argoverse/imgs_artificial/scene_PIT _000003_00.npy")
        # print(x.shape)
