import math

import curvox.cloud_conversions
import curvox.pc_vox_utils
import numpy as np
import pcl
import PyKDL
import tf_conversions
import torch as tc
import yaml
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

patch_size = 40


class ShapeCompletionDataset(Dataset):
    def __init__(self, yaml_file_path, models='train_models_train_views'):
        """
        Args:
        """
        self.models = models
        self.file_path = yaml_file_path
        with open(yaml_file_path, "r") as stream:
            try:
                self.data = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.length = len(self.data[models])

    def set_model(self, model):
        print("Setting model to " + model)
        self.models = model
        self.length = len(self.data[model])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_data = self.data[self.models][idx][0]
        output_data = self.data[self.models][idx][1]
        x = pcl.load(input_data).to_array().astype(int)
        x_mask = (x[:, 0], x[:, 1], x[:, 2])
        y = pcl.load(output_data).to_array().astype(int)
        y_mask = (y[:, 0], y[:, 1], y[:, 2])
        voxel_x = np.zeros((patch_size, patch_size, patch_size, 1),
                           dtype=np.float32)
        voxel_y = np.zeros((patch_size, patch_size, patch_size, 1),
                           dtype=np.float32)
        voxel_y[y_mask] = 1
        voxel_x[x_mask] = 1
        sample = {'pc': voxel_x, 'truth': voxel_y}
        return sample

    def get_item_name(self, string):
        string = string[:-6]
        string_list = string.split("/")
        idx = -1
        for i, j in enumerate(string_list):
            if j == "ycb" or j == "grasp_database":
                idx = i
        return string_list[idx + 1] + string_list[-1]


class trainDataSet(ShapeCompletionDataset):
    def __init__(self, yaml_file_path, models='train_models_train_views'):
        super(trainDataSet, self).__init__(yaml_file_path, models)


class testDataSet(ShapeCompletionDataset):
    def __init__(self, yaml_file_path, models='train_models_train_views'):
        super(testDataSet, self).__init__(yaml_file_path, models)

    def __getitem__(self, idx):
        input_data = self.data[self.models][idx][0]
        output_data = self.data[self.models][idx][1]
        input_pc = input_data[:-5] + "pc.pcd"
        model_pose = input_data[:-5] + "model_pose.npy"
        y = pcl.load(output_data).to_array().astype(int)
        y_mask = (y[:, 0], y[:, 1], y[:, 2])
        voxel_y = np.zeros((patch_size, patch_size, patch_size, 1),
                           dtype=np.float32)

        model_pose = np.load(model_pose)

        pc = pcl.load(input_pc).to_array()

        partial_vox = curvox.pc_vox_utils.pc_to_binvox_for_shape_completion(
            points=pc[:, 0:3], patch_size=40)
        voxel_x = np.zeros((patch_size, patch_size, patch_size, 1),
                           dtype=np.float32)
        voxel_x[:, :, :, 0] = partial_vox.data
        object_name = self.get_item_name(input_data)
        return voxel_x, voxel_y, pc, object_name, model_pose


class deterRandomSampler(Sampler):
    def __init__(self, data_source, seed):
        self.data_source = data_source
        self.seed = seed

    def __iter__(self):
        tc.manual_seed(self.seed)
        return iter(tc.randperm(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)
