from shape_reconstruction.utils import (file_utils, shape_completion_utils,
                                        argparser)

import copy
import os
import re

import numpy as np
from curvox import mesh_comparisons


def fix_mesh(mesh):
    f = open(mesh, "r")
    lines = f.readlines()
    f.close()
    correct = []
    j = 0
    rewrite = False
    for line in lines:
        temp = line.rstrip().split(" ")
        temp.pop(0)
        if len(temp) != len(set(temp)):
            rewrite = True
            j += 1
            continue
        correct.append(line)
    if rewrite:
        temp = correct[6].rstrip().split(" ")
        temp[-1] = str(int(correct[6].rstrip().split(" ")[-1]) - j)
        correct[6] = (" ").join(temp) + '\r\n'
        f = open(mesh, "w")
        for line in correct:
            f.write(line)
        f.close()


def compute_hausdorff_distance(first_mesh, second_mesh):
    hausdorff_distance = mesh_comparisons.hausdorff_distance_bi(
        first_mesh, second_mesh)
    if hausdorff_distance is not None:
        hausdorff_distance = hausdorff_distance["mean_distance"]
    else:
        hausdorff_distance = -1

    return hausdorff_distance


def compute_jaccard_similarity(first_mesh,
                               second_mesh,
                               voxel_resolution_to_use=80):
    jaccard_similarity = -1
    try:
        jaccard_similarity = mesh_comparisons.jaccard_similarity(
            first_mesh, second_mesh, grid_size=voxel_resolution_to_use)
    except IOError as e:
        print("Error computing jaccard sim between " + first_mesh +
              " and                   " + second_mesh)
        print(e)
        print("This error can occur if the program is run over ssh")

    return jaccard_similarity


def list2str(l):
    string = ""
    for i in l:
        try:
            for j in i:  # loop over all elements in the tuple
                string += str(j) + " "
            string += "\n"
        except TypeError:
            print("Problems iterating over " + str(i))
    return string.rstrip()


def dict2str(l):
    string = ""
    for i in l:
        try:
            string += i + " " + str(l[i])
            string += "\n"
        except TypeError:
            print("Problems iterating over " + str(i))
    return string.rstrip()


def save_data(data, folder, file_name):
    if type(data) is list:
        data = list2str(data)
    elif type(data) is dict:
        data = dict2str(data)
    data_file = file_utils.create_file(folder, file_name)
    file_utils.log_data_to_file(data, data_file)


def calculate_mean_of_values_in_list(list):
    return sum(list) / len(list)


def save_reconstruction_results(jaccard_similarity, hausdorff_distance, folder,
                                test_case, shape_completion_method):
    folder_to_save_results_to = folder + shape_completion_method + "/" + test_case + "/"
    print(folder_to_save_results_to)
    save_data(jaccard_similarity, folder_to_save_results_to,
              "jaccard_similarity_for_all_meshes.txt")
    save_data(hausdorff_distance, folder_to_save_results_to,
              "hausdorff_distance_for_all_meshes.txt")
    save_data(calculate_mean_of_values_in_list(jaccard_similarity.values()),
              folder_to_save_results_to, "mean_jaccard_similarity.txt")
    save_data(calculate_mean_of_values_in_list(hausdorff_distance.values()),
              folder_to_save_results_to, "mean_hausdorff_distance.txt")


def test_shape_reconstruction(ground_truth_meshes, shape_completed_meshes):

    jaccard_similarity_for_all_meshes = {}
    hausdorff_distance_for_all_meshes = {}
    for shape_completed_mesh in shape_completed_meshes:
        ground_truth_mesh = shape_completion_utils.get_ground_truth_mesh_corresponding_to_shape_completed_mesh(
            ground_truth_meshes, shape_completed_mesh)
        fix_mesh(ground_truth_mesh)
        jaccard_similarity = compute_jaccard_similarity(
            ground_truth_mesh, shape_completed_mesh)
        hausdorff_distance = compute_hausdorff_distance(
            ground_truth_mesh, shape_completed_mesh)
        jaccard_similarity_for_all_meshes[
            shape_completed_mesh] = jaccard_similarity
        hausdorff_distance_for_all_meshes[
            shape_completed_mesh] = hausdorff_distance
    return jaccard_similarity_for_all_meshes, hausdorff_distance_for_all_meshes


def compare_shape_completed_meshes_to_ground_truth_meshes(
    shape_completion_method,
    ground_truth_mesh_file_paths,
    shape_completed_mesh_file_paths,
    folder_to_save_results="/tmp/"):
    ground_truth_meshes, shape_completed_meshes = shape_completion_utils.load_ground_truth_and_shape_completed_meshes(
        ground_truth_mesh_file_paths, shape_completed_mesh_file_paths)
    for test_case, shape_completed_mesh_files in shape_completed_meshes.items(
    ):
        jaccard_similarity, hausdorff_distance = test_shape_reconstruction(
            ground_truth_meshes, shape_completed_mesh_files)
        folder_to_save_results + shape_completion_method + "/" + test_case
        save_reconstruction_results(jaccard_similarity, hausdorff_distance,
                                    folder_to_save_results, test_case,
                                    shape_completion_method)


if __name__ == "__main__":
    parser = argparser.reconstruction_parser()
    args = parser.parse_args()
    compare_shape_completed_meshes_to_ground_truth_meshes(
        args.completion_method, args.ground_truth_data_dir,
        args.shape_completed_data_dir, args.save_dir)
