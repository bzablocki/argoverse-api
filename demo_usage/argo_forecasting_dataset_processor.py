import sys
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pickle
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
import matplotlib
matplotlib.use('Agg')
# %%


class ForecastingDatasetProcessor():
    def __init__(self, root_dir=None, afl=None, avm=None):
        self.root_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-forecasting/train/data' if root_dir is None else root_dir
        self.pickle_path = "/media/bartosz/hdd1TB/workspace_hdd/SS-LSTM/data/argoverse/ss_lstm_format_argo_forecasting_v1{validation_flag}.pickle"
        self.pickle_cache_path = "/media/bartosz/hdd1TB/workspace_hdd/SS-LSTM/data/argoverse/cacheio/ss_lstm_format_argo_forecasting_v1_{range}.pickle"
        self.img_dataset_dir = "/media/bartosz/hdd1TB/workspace_hdd/SS-LSTM/data/argoverse/imgs_ds_forecasting"
        self.obs, self.pred = 20, 30
        self.afl = ArgoverseForecastingLoader(self.root_dir) if afl is None else afl
        self.avm = ArgoverseMap() if avm is None else avm
        self.scene_input, self.social_input, self.person_input, self.expected_output = None, None, None, None
        self.total_nb_of_segments = len(self.afl)
        # print('Loaded total number of sequences:', len(afl))

    @staticmethod
    def get_plot(map_range, return_ready_to_save=False):
        my_dpi = 96.0
        if return_ready_to_save:
            fig = plt.figure(figsize=(72 / my_dpi, 72 / my_dpi), dpi=my_dpi)
            # fig = plt.figure(figsize=(500 / my_dpi, 500 / my_dpi), dpi=my_dpi)
        else:
            fig = plt.figure(figsize=(5, 5), dpi=my_dpi)

        ax = fig.add_axes([0., 0., 1., 1.])
        ax.set_xlim([map_range[0], map_range[1]])
        ax.set_ylim([map_range[2], map_range[3]])

        if return_ready_to_save:
            fig.tight_layout(pad=0)
            ax.axis('off')

        return fig, ax

    @staticmethod
    def rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    @staticmethod
    def get_map_range(point, r=40):
        [xcenter, ycenter] = point
        xmin = xcenter - r
        xmax = xcenter + r
        ymin = ycenter - r
        ymax = ycenter + r

        return [xcenter, ycenter], [xmin, xmax, ymin, ymax]

    def add_map(self, ax, map_range, city_name):
        map_polygons = self.avm.find_local_lane_polygons(map_range, city_name)

        # ax.set_facecolor("black")
        for i in range(0, len(map_polygons)):
            poly = map_polygons[i]
            ax.add_patch(Polygon(poly[:, 0:2], facecolor="black", alpha=1))

        return ax

    @staticmethod
    def add_trajectory(ax, agent_obs_traj, agent_pred_traj=None, color="g"):
        ax.scatter(agent_obs_traj[:, 0], agent_obs_traj[:, 1], c=color, zorder=20)
        if agent_pred_traj is not None:
            ax.scatter(agent_pred_traj[:, 0], agent_pred_traj[:, 1], c='b', zorder=20)
        return ax

    def _get_circle_occupancy_map_single(self, target_object, traj_start_idx):
        neighborhood_radius, grid_radius, grid_angle = 32, 4, 45

        object_starting_frame = target_object['start']
        other_vehicles = target_object['other_vehicles']
        map_range = target_object['map_range']
        [xmin, xmax, ymin, ymax] = map_range

        width, height = xmax - xmin, ymax - ymin
        neighborhood_bound = neighborhood_radius / (min(width, height) * 1.0)
        grid_bound = grid_radius / (min(width, height) * 1.0)
        o_map = np.zeros((self.observed_frame_num, int(neighborhood_radius / grid_radius), int(360 / grid_angle)))

        for other_vehicle_id, (uuid_other_vehicle, other_vehicle) in enumerate(other_vehicles.items()):
            positions = other_vehicle['positions10Hz']
            other_positions = self._normalize_positions(positions, map_range)

            for frame_id, pos in enumerate(other_positions):
                relative_frame_nb = int(pos[0] - object_starting_frame)

                # process only other vehicles frames that happend within the obs period
                if traj_start_idx <= relative_frame_nb < traj_start_idx + self.observed_frame_num:
                    [current_x, current_y] = self._normalize_positions(target_object['positions10Hz'][relative_frame_nb, 1:3], map_range)
                    [other_x, other_y] = pos[1:3]
                    other_distance = np.sqrt((other_x - current_x) ** 2 + (other_y - current_y) ** 2)
                    angle = self._cal_angle(current_x, current_y, other_x, other_y)

                    if other_distance < neighborhood_bound:
                        cell_x, cell_y = int(np.floor(other_distance / grid_bound)), int(np.floor(angle / grid_angle))

                        o_map[relative_frame_nb - traj_start_idx, cell_x, cell_y] += 1
        return np.reshape(o_map, (self.observed_frame_num, -1))

    def get_city_name(self, forecasting_entry):
        seq_df = forecasting_entry.seq_df
        return seq_df['CITY_NAME'].iloc[0]

    def get_other_vehicles(self, forecasting_entry):
        others_in_timestamps = []

        seq_df = forecasting_entry.seq_df
        timestamps = np.unique(seq_df["TIMESTAMP"].values)
        others_df = seq_df[seq_df["OBJECT_TYPE"] == "OTHERS"]

        for t in timestamps:
            others_in_t_df = others_df[others_df["TIMESTAMP"] == t]

            others_x = others_in_t_df["X"]
            others_y = others_in_t_df["Y"]
            others_positions = np.column_stack((others_x, others_y))

            others_in_timestamps.append(others_positions)

        return others_in_timestamps

    def get_circle_occupancy_map_old(self, target_object, traj_start_idx):
        neighborhood_radius, grid_radius, grid_angle = 32, 4, 45

        object_starting_frame = target_object['start']
        other_vehicles = target_object['other_vehicles']
        map_range = target_object['map_range']
        [xmin, xmax, ymin, ymax] = map_range

        width, height = xmax - xmin, ymax - ymin
        neighborhood_bound = neighborhood_radius / (min(width, height) * 1.0)
        grid_bound = grid_radius / (min(width, height) * 1.0)
        o_map = np.zeros((self.observed_frame_num, int(neighborhood_radius / grid_radius), int(360 / grid_angle)))

        for other_vehicle_id, (uuid_other_vehicle, other_vehicle) in enumerate(other_vehicles.items()):
            positions = other_vehicle['positions10Hz']
            other_positions = self._normalize_positions(positions, map_range)

            for frame_id, pos in enumerate(other_positions):
                relative_frame_nb = int(pos[0] - object_starting_frame)

                # process only other vehicles frames that happend within the obs period
                if traj_start_idx <= relative_frame_nb < traj_start_idx + self.observed_frame_num:
                    [current_x, current_y] = self._normalize_positions(target_object['positions10Hz'][relative_frame_nb, 1:3], map_range)
                    [other_x, other_y] = pos[1:3]
                    other_distance = np.sqrt((other_x - current_x) ** 2 + (other_y - current_y) ** 2)
                    angle = self._cal_angle(current_x, current_y, other_x, other_y)

                    if other_distance < neighborhood_bound:
                        cell_x, cell_y = int(np.floor(other_distance / grid_bound)), int(np.floor(angle / grid_angle))

                        o_map[relative_frame_nb - traj_start_idx, cell_x, cell_y] += 1
        return np.reshape(o_map, (self.observed_frame_num, -1))

    def _cal_angle(self, current, next, other_x, other_y):
        [current_x, current_y] = current
        [next_x, next_y] = next
        p0 = [other_x, other_y]
        p1 = [current_x, current_y]
        # p2 = [current_x + 0.1, current_y]
        p2 = [next_x, next_y]
        v0 = np.array(p0) - np.array(p1)
        v1 = np.array(p2) - np.array(p1)
        angle_degree = np.degrees(np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1)))
        return angle_degree

    @staticmethod
    def is_within(map_range, pos):
        [xmin, xmax, ymin, ymax] = map_range
        return xmin < pos[0] < xmax and ymin < pos[1] < ymax

    def get_circle_occupancy_map(self, other_vehicles, obs_traj, map_range, frame_number=None, ax=None):
        grid_radius = 4
        neighborhood_bound, grid_angle = grid_radius * 8, 45
        # [xmin, xmax, ymin, ymax] = map_range
        # width, height = xmax - xmin, ymax - ymin
        # neighborhood_bound = neighborhood_radius / (min(width, height) * 1.0)
        # grid_bound = grid_radius / (min(width, height) * 1.0)
        o_map = np.zeros((self.obs, int(neighborhood_bound / grid_radius), int(360 / grid_angle)))
        # print(f"o_map shape: {o_map.shape}")

        rang = range(self.obs) if frame_number is None else range(frame_number, frame_number + 1)

        for frame_nb in rang:
            for other_vehicle in other_vehicles[frame_nb]:
                [current_x, current_y] = obs_traj[frame_nb]
                [other_x, other_y] = other_vehicle
                other_distance = np.sqrt((other_x - current_x) ** 2 + (other_y - current_y) ** 2)

                if frame_nb + 1 < self.obs:  # special case: last frame. Use two frames before to calculate angels
                    # [next_x, next_y] = obs_traj[frame_nb+1] if frame_nb+1 < self.obs else [current_x + 0.1, current_y]
                    angle = self._cal_angle(obs_traj[frame_nb], obs_traj[frame_nb + 1], other_x, other_y)
                else:
                    angle = self._cal_angle(obs_traj[frame_nb - 1], obs_traj[frame_nb], other_x, other_y)

                cell_x, cell_y = int(np.floor(other_distance / grid_radius)), int(np.floor(angle / grid_angle))

                is_inbound = other_distance < neighborhood_bound
                if ax is not None and self.is_within(map_range, [other_x, other_y]):  # and is_inbound:
                    text_pos_x, text_pos_y = other_x + np.random.randint(-10, 10), other_y + np.random.randint(-10, 10)
                    ax.arrow(other_x, other_y, text_pos_x - other_x, text_pos_y - other_y, color="b")
                    ax.text(text_pos_x, text_pos_y, f"a:{int(angle)}, d:{int(other_distance)} -> {cell_x}, {cell_y} -> {is_inbound}", color="magenta", zorder=200)
                    ax.scatter(other_x, other_y, c="gray")
                    # print(f"{other_distance} < {neighborhood_bound} -> {other_distance < neighborhood_bound}")

                if is_inbound:
                    o_map[frame_nb, cell_x, cell_y] += 1

        return o_map, ax

    # def get_data(self):
    #     if self.scene_input is None or self.social_input is None or self.person_input is None or self.expected_output:
    #         self.scene_input, self.social_input, self.person_input, self.expected_output = self.generate_network_data(dev_mode=False)
    #     return self.scene_input, self.social_input, self.person_input, self.expected_output

    def save_data_to_pickle(self, data, is_validation=False, range=None):
        if range is None:
            path = self.pickle_path.format(validation_flag="_val") if is_validation else self.pickle_path.format(validation_flag="")
        else:
            path = self.pickle_cache_path.format(range=f"{range[0]}_{range[-1]}")

        pickle_out = open(path, "wb")
        pickle.dump(data, pickle_out, protocol=2)
        pickle_out.close()
        print("Saved to pickle {}".format(path))

    def get_social_input_single(self, forecasting_entry, agent_obs_traj, map_range):
        other_vehicles = self.get_other_vehicles(forecasting_entry)
        o_map, _ = self.get_circle_occupancy_map(other_vehicles, agent_obs_traj, map_range)
        return o_map

    def get_scene_input_single(self, map_range, seq_id, city_name):
        img_path = os.path.join(self.img_dataset_dir, 'scene_{}.npy'.format(seq_id))
        if not os.path.exists(img_path):
            fig, ax = self.get_plot(map_range, return_ready_to_save=True)
            ax = self.add_map(ax, map_range, city_name)
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            data = np.rot90(data.transpose(1, 0, 2))
            data = self.rgb2gray(data)
            data = 1 - data
            np.save(img_path, data)
            plt.close(fig)
        #     print(f"Scene saved to   {img_path}")
        # else:
        #     print(f"Scene existed in {img_path}")

        return img_path

    def generate_network_data(self, dev_mode=True, save_to_pickle=False, desired_range=None):
        scene_input, social_input, person_input, expected_output = np.zeros((0, 1)), np.zeros((0, self.obs, 64)), np.zeros((0, self.obs, 2)), np.zeros((0, self.pred, 2))

        full_trajectories_len = self.total_nb_of_segments
        if desired_range is not None:
            selected_range = desired_range
        else:
            selected_range = range(0, 10) if dev_mode else range(full_trajectories_len)

        print(f"Processing range {selected_range}")

        for seq_id in selected_range:
            forecasting_entry = self.afl[seq_id]
            traj = forecasting_entry.agent_traj
            agent_obs_traj = traj[:self.obs]
            agent_pred_traj = traj[self.obs:]
            # city_name = forecasting_entry.city
            city_name = self.get_city_name(forecasting_entry)
            center, map_range = self.get_map_range(agent_obs_traj[-1], r=40)
            # if dev_mode:  # not
            # if not dev_mode:  # not
            person_input = np.vstack((person_input, np.expand_dims(agent_obs_traj, axis=0)))
            expected_output = np.vstack((expected_output, np.expand_dims(agent_pred_traj, axis=0)))

            social_input_single = self.get_social_input_single(forecasting_entry, agent_obs_traj, map_range)
            social_input = np.vstack((social_input, np.reshape(social_input_single, (1, self.obs, -1))))

            scene_input_single = self.get_scene_input_single(map_range, seq_id, city_name)
            scene_input = np.vstack((scene_input, np.reshape(np.array(scene_input_single).astype('object'), (1, 1))))

            # log checkpoint
            log_every = 100
            if (seq_id) % log_every == 0:
                print("Progress: {}% // ".format(int((seq_id - selected_range[0]) / (selected_range[-1] - selected_range[0]) * 100)))

        print("Generated shapes: {}".format((scene_input.shape, social_input.shape, person_input.shape, expected_output.shape)))
        if save_to_pickle:
            self.save_data_to_pickle([scene_input, social_input, person_input, expected_output], range=desired_range)

        return scene_input, social_input, person_input, expected_output

    def get_cache_path(self, desired_range):
        return self.pickle_cache_path.format(range=f"{desired_range[0]}_{desired_range[-1]}")

    def get_range_from_segment(self, segment_nb):
        split_every = 20000
        total_nb_of_segments = self.total_nb_of_segments
        range_start = segment_nb * split_every
        if range_start >= total_nb_of_segments:
            print("Alrighty, that's too much")
            return 0

        range_end = min(range_start + split_every, total_nb_of_segments)
        desired_range = range(range_start, range_end)
        return desired_range

    def merge_cache(self):
        scene_input, social_input, person_input, expected_output = np.zeros((0, 1)), np.zeros((0, self.obs, 64)), np.zeros((0, self.obs, 2)), np.zeros((0, self.pred, 2))
        for i in range(0, 11):
            pckl_path = self.get_cache_path(self.get_range_from_segment(i))
            pickle_in = open(pckl_path, "rb")
            [scene_input_pckl, social_input_pckl, person_input_pckl, expected_output_pckl] = pickle.load(pickle_in)

            if i == 10:  # split last segment and use for val dataset
                scene_input_pckl, scene_input_val, social_input_pckl, social_input_val, person_input_pckl, person_input_val, expected_output_pckl, expected_output_val = train_test_split(
                    scene_input_pckl, social_input_pckl, person_input_pckl, expected_output_pckl, test_size=0.2, random_state=42)
                print("Generated val shapes: {}".format((scene_input_val.shape, social_input_val.shape, person_input_val.shape, expected_output_val.shape)))
                self.save_data_to_pickle([scene_input_val, social_input_val, person_input_val, expected_output_val], is_validation=True, range=None)

            scene_input = np.vstack((scene_input, scene_input_pckl))
            social_input = np.vstack((social_input, social_input_pckl))
            person_input = np.vstack((person_input, person_input_pckl))
            expected_output = np.vstack((expected_output, expected_output_pckl))
            print(f"Segment {i} done")

        print("Generated shapes: {}".format((scene_input.shape, social_input.shape, person_input.shape, expected_output.shape)))
        self.save_data_to_pickle([scene_input, social_input, person_input, expected_output], is_validation=False, range=None)


def main():
    # ''''''''''''''''
    # root_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-forecasting/train/data'
    # afl = ArgoverseForecastingLoader(root_dir)
    # avm = ArgoverseMap()
    # ''''''''''''''''
    ''' Select which segment to cache '''
    if len(sys.argv) >= 3 and sys.argv[1] == "split_to_cache" and sys.argv[2].isdigit():
        fdp = ForecastingDatasetProcessor()
        desired_range = fdp.get_range_from_segment(int(sys.argv[2]))
        final_cache_path = fdp.get_cache_path(desired_range)
        if os.path.exists(final_cache_path):
            print(f"Cache {final_cache_path} already exists")
            return 0

        print(f"Generating cache from {range_start} to {range_end}")
        fdp.generate_network_data(dev_mode=False, save_to_pickle=True, desired_range=desired_range)

    if len(sys.argv) >= 2 and sys.argv[1] == "merge_cache":
        print("Merging cache")
        fdp = ForecastingDatasetProcessor()
        fdp.merge_cache()
    else:
        fdp = ForecastingDatasetProcessor()
        fdp.generate_network_data(dev_mode=False, save_to_pickle=True, desired_range=range(20000, 20010))


if __name__ == '__main__':
    main()
