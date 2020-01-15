# coding: utf-8

import os
import yaml
import random
import pcl
import numpy as np
import sys
import argparse
from random import shuffle
from shape_reconstruction.utils import file_utils


def get_train_and_test_model_names(dataset_seed, use_default_train_test_split):
    if use_default_train_test_split:
        model_names = file_utils.read_yaml_file(
            "default_train_test_split.yaml")
        train_model_names = model_names["grasp_database_objects"][
            'train_model_names']
        holdout_model_names = model_names["grasp_database_objects"][
            'holdout_model_names']
        train_model_names += model_names["ycb_objects"]['train_model_names']
        holdout_model_names += model_names["ycb_objects"][
            'holdout_model_names']

    else:
        model_names = file_utils.read_yaml_file("objects.yaml")
        ycb_train_test_split = remove_a_percentage_of_items_from_a_list(
            model_names['ycb_objects'], 0.25)
        grasp_database_train_test_split = remove_a_percentage_of_items_from_a_list(
            model_names['grasp_database_objects'], 0.25)
        train_model_names = ycb_train_test_split[
            0] + grasp_database_train_test_split[0]
        holdout_model_names = ycb_train_test_split[
            1] + grasp_database_train_test_split[1]
    return train_model_names, holdout_model_names


def load_and_verify_input_data(
    path_to_data,
    model_names,
    patch_size,
):

    data = []
    for model in model_names:
        model_data = model + "/pointclouds"
        data_folders = [
            path_to_data + "grasp_database/" + model_data,
            path_to_data + "ycb/" + model_data
        ]
        if os.path.exists(data_folders[0]):
            absolute_data_dir_path = os.path.abspath(data_folders[0])
            for mfile in os.listdir(absolute_data_dir_path):
                if "x.pcd" in mfile:
                    x = absolute_data_dir_path + "/" + mfile
                    y = x.replace("x.pcd", "y.pcd")
                    if verify_example(x, y, patch_size):
                        data.append((x, y))
        elif os.path.exists(data_folders[1]):
            absolute_data_dir_path = os.path.abspath(data_folders[1])
            for mfile in os.listdir(absolute_data_dir_path):
                if "x.pcd" in mfile:
                    x = absolute_data_dir_path + "/" + mfile
                    y = x.replace("x.pcd", "y.pcd")
                    if verify_example(x, y, patch_size):
                        data.append((x, y))
        else:
            print("Folder containing train and test data for " + model +
                  " does not exist")
    return data


def remove_a_percentage_of_items_from_a_list(list_to_remove_items_from,
                                             percentage_to_remove):
    shuffle(list_to_remove_items_from)
    count = int(len(list_to_remove_items_from) * percentage_to_remove)
    if not count: return []  # edge case, no elements removed
    list_to_remove_items_from[
        -count:], list_of_removed_items = [], list_to_remove_items_from[
            -count:]
    return list_to_remove_items_from, list_of_removed_items


def build(dataset_type, path_to_data, dataset_seed,
          use_default_train_test_split):
    train_model_names, holdout_model_names = get_train_and_test_model_names(
        dataset_seed, use_default_train_test_split)
    patch_size = 40
    if dataset_type == "debug":
        train_model_names, _ = remove_a_percentage_of_items_from_a_list(
            train_model_names, 0.9)
        holdout_model_names, _ = remove_a_percentage_of_items_from_a_list(
            holdout_model_names, 0.9)

    dataset = {}
    dataset["train_model_names"] = train_model_names
    dataset["holdout_model_names"] = holdout_model_names
    dataset["patch_size"] = patch_size
    train_models_train_views = []
    train_models_holdout_views = []
    holdout_models_holdout_views = []
    print("Starting building training models")
    train_models_train_views = load_and_verify_input_data(
        path_to_data, train_model_names, patch_size)
    train_models_train_views, train_models_holdout_views = remove_a_percentage_of_items_from_a_list(
        train_models_train_views, 0.8)

    print("Starting building holdout models")
    holdout_models_holdout_views = load_and_verify_input_data(
        path_to_data, holdout_model_names, patch_size)

    dataset['train_models_train_views'] = train_models_train_views
    dataset['train_models_holdout_views'] = train_models_holdout_views
    dataset['holdout_models_holdout_views'] = holdout_models_holdout_views
    dataset_folder = os.getcwd() + "/build_datasets/"
    dataset_file = dataset_type + "_dataset.yaml"
    file_utils.create_file(dataset_folder, dataset_file)
    with open(dataset_folder + dataset_file, "w") as outfile:
        yaml.dump(dataset, outfile, default_flow_style=True)

    return dataset_folder + dataset_file


def verify_example(x_filepath, y_filepath, patch_size):

    success = True

    x = pcl.load(x_filepath).to_array().astype(int)
    x_mask = (x[:, 0], x[:, 1], x[:, 2])
    y = pcl.load(y_filepath).to_array().astype(int)
    y_mask = (y[:, 0], y[:, 1], y[:, 2])

    voxel_x = np.zeros((patch_size, patch_size, patch_size, 1),
                       dtype=np.float32)
    voxel_y = np.zeros((patch_size, patch_size, patch_size, 1),
                       dtype=np.float32)

    voxel_y[y_mask] = 1
    voxel_x[x_mask] = 1

    x_count = np.count_nonzero(voxel_x)

    overlap = voxel_x * voxel_y
    overlap_count = np.count_nonzero(overlap)

    # this should be high, almost all points should be in gt
    percent_overlap = float(overlap_count) / float(x_count)

    overlap_threshold = 0.8
    if percent_overlap < overlap_threshold:
        print("pcl_viewer " + x_filepath + " has overlap < " +
              str(overlap_threshold))
        success = False

    pts_threshold = 100
    if x_count < pts_threshold:
        print("pcl_viewer " + x_filepath + " has less than " +
              str(pts_threshold) + " points !!")
        success = False

    return success


def verify(dataset_file_path):

    Success = True
    problems = []
    dataset_type = yaml.load(open(dataset_file_path, 'r'))
    for key in [
            'train_models_train_views', 'train_models_holdout_views',
            'holdout_models_holdout_views'
    ]:
        for x_filepath, y_filepath in dataset_type[key]:

            patch_size = dataset_type["patch_size"]
            if not verify_example(x_filepath, y_filepath, patch_size):
                Success = False
                problems.append(x_filepath)

    return Success, problems


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def checkPositive(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value" % value)
    return ivalue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        'Build the datasets that are needed for training and testing')
    parser.add_argument(
        '--dataset-type',
        type=str,
        choices=["train_test", "debug"],
        default="train_test",
        nargs='?',
        help=
        'What type of dataset you want to create. The debug dataset is just a much smaller version of the train dataset that is quicker to load and should be used for debugging the network'
    )
    parser.add_argument('--path-to-dataset',
                        type=str,
                        default="./training_data/",
                        help='Path to the train and test data folder')
    parser.add_argument(
        '--dataset-seed',
        type=checkPositive,
        default=100,
        help=
        'A random seed to split the ycb and grasp dataset objects into a training and holdout set'
    )
    parser.add_argument(
        '--use-default-split',
        type=str2bool,
        default=True,
        help=
        'Set this to true if you want to use the default train-test split as specified in default_train_test_split.yaml'
    )

    args = parser.parse_args()

    print("BUILDING DATASET " + args.dataset_type)
    dataset_file = build(args.dataset_type, args.path_to_dataset,
                         args.dataset_seed, args.use_default_split)
    print("VERIFYING DATASET " + args.dataset_type)
    Success, problems = verify(dataset_file)
    if Success:
        print("DATASET BUILD AND VERIFIED")
    else:
        print(problems)
