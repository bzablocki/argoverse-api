# %%
from argoverse.map_representation.map_api import ArgoverseMap
import numpy as np
from demo_usage.visualize_30hz_benchmark_data_on_map import DatasetOnMapVisualizer
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from matplotlib.patches import Polygon
import pickle
import logging
import os
# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# %%
class ArgoverseUtils():

    def __init__(self, argoverse_data, argoverse_map):
        self.argoverse_data = argoverse_data
        self.argoverse_map = argoverse_map
        self.img_dataset_dir = "/media/bartosz/hdd1TB/workspace_hdd/SS-LSTM/data/argoverse/imgs"

    def get_map_range(self, idx):
        xcenter, ycenter, _ = self.argoverse_data.get_pose(idx).translation
        r = 25
        xmin = xcenter - r
        xmax = xcenter + r
        ymin = ycenter - r
        ymax = ycenter + r

        return [xcenter, ycenter], [xmin, xmax, ymin, ymax]

    @staticmethod
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

    @staticmethod
    def is_within(pos, map_range):
        [xmin, xmax, ymin, ymax] = map_range
        return xmin <= pos[1] <= xmax and ymin <= pos[2] <= ymax

    def trim_path_to_visible(self, positions):
        valid_positions = []
        positions = np.array(positions)

        map_range = None
        ego_pos = None
        starting_position_found = False

        for pos in positions:
            if not starting_position_found:
                ego_pos, map_range = self.get_map_range(int(pos[0]))

            if self.is_within(pos, map_range):
                valid_positions.append(pos)
                starting_position_found = True

        return ego_pos, map_range, np.array(valid_positions)

    def trim_and_interpolate_object(self, target_object):
        positions = target_object['positions10Hz']
        ego_pos, map_range, positions = self.trim_path_to_visible(positions)

        if positions.size:
            positions = self.interpolate_missing_frames(positions)

            target_object['start'] = int(positions[0][0])
            target_object['stop'] = int(positions[-1][0])
            target_object['positions10Hz'] = positions
            target_object['map_range'] = map_range
            target_object['ego_pos'] = ego_pos

            return target_object

        return None

    def add_other_vehicles(self, target_uuid, target_object, other_objects):
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
                other_postitions = other_postitions[(other_postitions[:, 0] >= target_start) & (other_postitions[:, 0] <= target_stop)]

                # iterate through all the positions
                for other_pos in other_postitions:

                    # check if ego_vehicle can actually see it
                    if self.is_within(other_pos, map_range):
                        # if the dict doesn't have an entry with the uuid, initialize it
                        if other_uuid not in other_objects_valid:
                            other_objects_valid[other_uuid] = dict()
                            other_objects_valid[other_uuid]['positions10Hz'] = None

                        # add the position entry if the ego vehicle can see the object
                        other_objects_valid[other_uuid]['positions10Hz'] = np.vstack(
                            (other_objects_valid[other_uuid]['positions10Hz'], other_pos)) if other_objects_valid[other_uuid]['positions10Hz'] is not None else np.reshape(other_pos, (1, -1))

        target_object['other_vehicles'] = other_objects_valid
        return target_object

    @staticmethod
    def get_plot(map_range, pix_to_pix_mapping=True):
        if pix_to_pix_mapping:
            my_dpi = 96.0  # screen constant, check here https://www.infobyip.com/detectmonitordpi.php
            fig = plt.figure(figsize=(72 / my_dpi, 72 / my_dpi), dpi=my_dpi)
            # fig = plt.figure(figsize=(48 / my_dpi, 48 / my_dpi), dpi=my_dpi)
            # fig = plt.figure(figsize=(0.25, 0.25), dpi=my_dpi)
        else:
            my_dpi = 96.0  # screen constant, check here https://www.infobyip.com/detectmonitordpi.php
            fig = plt.figure(figsize=(48 / my_dpi, 48 / my_dpi), dpi=my_dpi)
            # fig = plt.figure(figsize=(1, 1))

        # fig.tight_layout(pad=0)
        ax = fig.add_axes([0., 0., 1., 1.])
        ax.set_xlim([map_range[0], map_range[1]])
        ax.set_ylim([map_range[2], map_range[3]])
        ax.axis('off')

        return fig, ax

    def add_map(self, ax, map_range):
        city_name = self.argoverse_data.city_name
        map_polygons = self.argoverse_map.find_local_lane_polygons(map_range, city_name)

        # ax.set_facecolor("black")
        for i in range(0, len(map_polygons)):
            poly = map_polygons[i]
            ax.add_patch(Polygon(poly[:, 0:2], facecolor="black", alpha=1))

        return ax

    @staticmethod
    def rgb2gray(rgb):

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray

    def add_img(self, target_object, uuid, save=False):
        map_range = target_object['map_range']

        fig, ax = self.get_plot(map_range, pix_to_pix_mapping=True)
        ax = self.add_map(ax, map_range)

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = np.rot90(data.transpose(1, 0, 2))
        data = self.rgb2gray(data)
        data = 1 - data
        # target_object['img'] = data
        # print(data.shape)
        img_path = os.path.join(self.img_dataset_dir, 'scene_{}.npy'.format(uuid))

        # if os.path.exists(img_path):
        #     print("WARNING: The file already exists")
        # if uuid == '60c089d4-10bf-4855-82c7-5ec02d0955ad':
        #     print("Saving in {}".format(img_path))
        #     print("shape {}".format(data.shape))
        #     plt.show()

        if save:
            np.save(img_path, data)
        target_object['img_path'] = img_path

        plt.close()

        return target_object

    def is_stationary(self, target_object, visualize=False):
        positions = target_object['positions10Hz']
        poss_x = positions[:, 1]
        poss_y = positions[:, 2]

        thr = 1.

        if visualize:
            map_range = target_object['map_range']
            # print(map_range)
            plt.scatter(poss_x, poss_y)
            plt.xlim(map_range[0], map_range[1])
            plt.ylim(map_range[2], map_range[3])
            plt.title(str(np.std(poss_x)) + " " + str(np.std(poss_y)) + " " + str((np.std(poss_x) < thr) and (np.std(poss_y) < thr)))
            plt.show()
        return (np.std(poss_x) < thr) and (np.std(poss_y) < thr)


class Argoverse():

    def __init__(self, tracking_dataset_dir, dataset_name=None, argoverse_map=None, argoverse_loader=None, save_imgs=False):
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)

        self.dataset_dir = tracking_dataset_dir
        self.am = ArgoverseMap() if argoverse_map is None else argoverse_map
        self.argoverse_loader = ArgoverseTrackingLoader(tracking_dataset_dir) if argoverse_loader is None else argoverse_loader
        self.dataset_prefix_name = dataset_name

        self.objects_from_to = self._get_objects_from_to()
        self.valid_target_objects = self._get_valid_target_objects(save_imgs=save_imgs)

    def _get_objects_from_to(self):
        objects_from_to = dict()

        for log_index in range(len(self.argoverse_loader.log_list)):
            log_id = self.argoverse_loader.log_list[log_index]
            argoverse_data = self.argoverse_loader[log_index]
            domv = DatasetOnMapVisualizer(self.dataset_dir, 'visualization_demo', use_existing_files=True, log_id=argoverse_data.current_log)

            for idxx in range(len(argoverse_data.lidar_timestamp_list)):
                objects = domv.log_timestamp_dict[log_id][argoverse_data.lidar_timestamp_list[idxx]]

                for i, obj in enumerate(objects):
                    if obj.obj_class_str == "VEHICLE":
                        if obj.track_uuid not in objects_from_to:
                            objects_from_to[obj.track_uuid] = dict()
                            objects_from_to[obj.track_uuid]['start'] = idxx
                            objects_from_to[obj.track_uuid]['log_index'] = log_index
                            objects_from_to[obj.track_uuid]['positions10Hz'] = []
                            objects_from_to[obj.track_uuid]['positions10Hz'].append(np.concatenate(([idxx], np.mean(obj.bbox_city_fr, axis=0)[0:2])))
                        else:
                            objects_from_to[obj.track_uuid]['stop'] = idxx
                            objects_from_to[obj.track_uuid]['positions10Hz'].append(np.concatenate(([idxx], np.mean(obj.bbox_city_fr, axis=0)[0:2])))
        return objects_from_to

    def _get_valid_target_objects(self, save_imgs=False):
        valid_target_objects = dict()
        for i, (uuid, target_object) in enumerate(self.objects_from_to.items()):
            utils = ArgoverseUtils(argoverse_data=self.argoverse_loader[target_object['log_index']], argoverse_map=self.am)
            # get a segment of the path that is visible by ego vehicle, and interpolate missing frames
            new_target_object = utils.trim_and_interpolate_object(target_object)
            if new_target_object is not None:
                new_target_object = utils.add_other_vehicles(uuid, new_target_object, self.objects_from_to)
                new_target_object = utils.add_img(new_target_object, uuid, save=save_imgs)
                new_target_object['is_stationary'] = utils.is_stationary(new_target_object)
                # add to the final dict
                valid_target_objects[uuid] = new_target_object

        return valid_target_objects

    def get_objects_from_to(self):
        return self.objects_from_to

    def get_valid_target_objects(self):
        return self.valid_target_objects

    def save_to_pickle(self, name=None):
        f = name if name is not None else "/media/bartosz/hdd1TB/workspace_hdd/SS-LSTM/data/argoverse/{}.pickle".format(self.dataset_prefix_name)
        pickle_out = open(f, "wb")
        pickle.dump(self.valid_target_objects, pickle_out, protocol=2)
        pickle_out.close()
        print("Saved to pickle {}".format(f))

def save_all_to_pickle():
    datasets = ["train1", "train2", "train3", "train4"]
    final_dict = {}
    for dataset in datasets:
        tracking_dataset_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-tracking/' + dataset
        ###################
        am = ArgoverseMap()
        argoverse_loader = ArgoverseTrackingLoader(tracking_dataset_dir)
        ###################
        argoverse = Argoverse(tracking_dataset_dir=tracking_dataset_dir, dataset_name=dataset, argoverse_map=am, argoverse_loader=argoverse_loader)
        final_dict.update(argoverse.valid_target_objects)
        print("Processed {}".format(dataset))

    f = "/media/bartosz/hdd1TB/workspace_hdd/SS-LSTM/data/argoverse/train1234_48x48.pickle"
    pickle_out = open(f, "wb")
    pickle.dump(final_dict, pickle_out, protocol=2)
    pickle_out.close()
    print("Saved to pickle {}".format(f))

def merge_dicts(d1, d2, d3):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    d1.update(d2)
    d1.update(d3)
    return d1


if __name__ == '__main__':
    pass
    # save_all_to_pickle()

    dataset = "train1"
    tracking_dataset_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-tracking/' + dataset
    ###################
    # am = ArgoverseMap()
    # argoverse_loader = ArgoverseTrackingLoader(tracking_dataset_dir)
    ###################
    argoverse = Argoverse(tracking_dataset_dir=tracking_dataset_dir, dataset_name=dataset, argoverse_map=am, argoverse_loader=argoverse_loader, save_imgs=True)
    # argoverse.save_to_pickle("/media/bartosz/hdd1TB/workspace_hdd/SS-LSTM/data/argoverse/{}_48x48.pickle".format(dataset))
    # %%
    data_dict = argoverse.valid_target_objects
    for i, (uuid, target_object)in enumerate(data_dict.items()):
        # if uuid == "85b331ea-cc36-48e9-a543-89ba6d2cbb30":
        #     print(i, "woww")
        if i < 243:
            continue
        if i > 243:
            break
        # if target_object["is_stationary"]:
        #     continue
        #
        print(target_object["img_path"])
        plt.imshow(np.load(target_object["img_path"]))


    # # idx = 6
    # idx = 103
    # # idx = 85
    # trajectories = argoverse.get_valid_target_objects()
    # for i, (uuid, target_object) in enumerate(trajectories.items()):
    #     if idx-1 < i < idx+1:
    #         if target_object["is_stationary"]:
    #             print(i, "stationary")
    #         print(target_object["img_path"])
    #         data = np.load(target_object["img_path"])
    #         print(data.shape)
    #         plt.imshow(data)
    #         positions = np.copy(target_object["positions10Hz"])
    #         map_range = target_object["map_range"]
    #         [xmin, xmax, ymin, ymax] = map_range
    #         positions[..., -2] = (positions[..., -2] - xmin) / (xmax - xmin)
    #         positions[..., -1] = (positions[..., -1] - ymin) / (ymax - ymin)
    #         positions[..., -2] = positions[..., -2] * 48
    #         positions[..., -1] = positions[..., -1] * 48
    #         print(positions[0])
    #         plt.scatter(positions[..., -2], positions[..., -1])
    #         plt.show()



    # print(positions['positions10Hz'])



# %%
# argoverse = Argoverse(tracking_dataset_dir=tracking_dataset_dir, dataset_name=dataset, argoverse_map=am, argoverse_loader=argoverse_loader)
# target_objects = argoverse.get_valid_target_objects()
# non_stationary_counter = 0
# for uuid, obj in target_objects.items():
#     if not obj['is_stationary']:
#         non_stationary_counter += 1
#
# print("Non stationary objects: {} ({:.2f}%)".format(
#     non_stationary_counter, non_stationary_counter / len(target_objects) * 100))
# %%
    # dataset = "val"
    # tracking_dataset_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-tracking/' + dataset
    # argoverse = Argoverse(tracking_dataset_dir=tracking_dataset_dir, dataset_name=dataset)
    # print("{} done".format(dataset))

    # argoverse.save_to_pickle(name="/media/bartosz/hdd1TB/workspace_hdd/SS-LSTM/data/argoverse/{}_gray.pickle".format(dataset))
# %%
# %%
#
# dataset = "train4"
# tracking_dataset_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-tracking/' + dataset
# argoverse = Argoverse(tracking_dataset_dir=tracking_dataset_dir, dataset_name=dataset)
# print("{} done".format(dataset))
#
# argoverse.save_to_pickle(name="/media/bartosz/hdd1TB/workspace_hdd/SS-LSTM/data/argoverse/{}_gray.pickle".format(dataset))


# # # %%
#
# img = np.load("/media/bartosz/hdd1TB/workspace_hdd/SS-LSTM/data/argoverse/imgs/scene_0a3b6731-24cd-4d23-879e-3312ee45d2ca.npy")
# print(img.shape)
# dataset = "train1"
# tracking_dataset_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-tracking/' + dataset
# argoverse1 = Argoverse(tracking_dataset_dir=tracking_dataset_dir, dataset_name=dataset)
# print("train1 done")
# argoverse1.save_to_pickle()
# # %%
# dataset = "train2"
# tracking_dataset_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-tracking/' + dataset
# argoverse2 = Argoverse(tracking_dataset_dir=tracking_dataset_dir, dataset_name=dataset)
# print("train2 done")
# argoverse2.save_to_pickle()
# # %%
#
# dataset = "train3"
# tracking_dataset_dir = '/media/bartosz/hdd1TB/workspace_hdd/datasets/argodataset/argoverse-tracking/' + dataset
# argoverse3 = Argoverse(tracking_dataset_dir=tracking_dataset_dir, dataset_name=dataset)
# print("train3 done")
# argoverse3.save_to_pickle()
# # %%
# all = merge_dicts(argoverse1.get_valid_target_objects(), argoverse2.get_valid_target_objects(), argoverse3.get_valid_target_objects())
# print("done")
# # %%
# f = "/media/bartosz/hdd1TB/workspace_hdd/SS-LSTM/data/argoverse/{}.pickle".format("train123")
# pickle_out = open(f, "wb")
# pickle.dump(all, pickle_out, protocol=2)
# pickle_out.close()
# print("Saved to pickle {}".format(f))
# # # %%
