import math
import os
import shutil
import subprocess
import tempfile

import curvox.cloud_to_mesh_conversions
import curvox.mesh_conversions
import curvox.pc_vox_utils
import numpy as np
import pcl
import plyfile
import PyKDL
import scipy.io
import tf_conversions
import torch as tc
import trimesh
import yaml
from curvox import mesh_conversions
from geometry_msgs import msg

import binvox_rw
import file_utils
import mcubes

FNULL = open(os.devnull, 'w')
patch_size = 40


def numpy_jaccard_similarity(a, b):
    '''
    Returns the number of pixels of the intersection of two voxel grids divided
    by the number of pixels in the union.
    The inputs are expected to be numpy 4D ndarrays in BXYZ format.
    '''
    return np.mean(
        np.sum(a * b, axis=(0, 1, 2)) / np.sum(
            (a + b) - a * b, axis=(0, 1, 2)))


def cuda_jaccard_similarity(a, b):
    '''
    Returns the number of pixels of the intersection of two voxel grids divided
    by the number of pixels in the union.
    The inputs are expected to be numpy 5D ndarrays in BZCXY format.
    '''
    numerator = a * b
    denominator = a + b - numerator
    return tc.mean((numerator.sum(2).sum(2).sum(2).sum(1)) /
                   (denominator.sum(2).sum(2).sum(2).sum(1)))


def get_test_cases(test_case):
    if test_case == "all":
        return [
            "train_models_train_views", "train_models_holdout_views",
            "holdout_models_holdout_views"
        ]
    else:
        return [test_case]


def set_pose_msg(position=[], orientation=[]):
    assert (len(position) == 0 or len(position) == 3)
    assert (len(orientation) == 0 or len(orientation) == 4)
    pose = msg.Pose()
    pose.orientation.w = 1
    if position:
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        if orientation:
            pose.orientation.w = orientation[0]
            pose.orientation.x = orientation[1]
            pose.orientation.y = orientation[2]
            pose.orientation.z = orientation[3]
    return pose


def scan_match(model1, model2):
    scan1 = trimesh.load_mesh(model1)
    scan2 = trimesh.load_mesh(model2)

    # (4, 4) float homogenous transfrom from scan2 to scan1
    scan1ToScan2, _ = scan2.register(scan1)
    return scan1ToScan2


def round_voxel_grid_to_0_and_1(voxel_grid):
    return np.round(voxel_grid)


def cnn_and_pc_to_mesh(observed_pc,
                       cnn_voxel,
                       filepath,
                       mesh_name,
                       model_pose,
                       log_pc=False,
                       pc_name=""):
    cnn_voxel = round_voxel_grid_to_0_and_1(cnn_voxel)
    temp_pcd_handle, temp_pcd_filepath = tempfile.mkstemp(suffix=".pcd")
    os.close(temp_pcd_handle)
    pcd = np_to_pcl(observed_pc)
    pcl.save(pcd, temp_pcd_filepath)

    partial_vox = curvox.pc_vox_utils.pc_to_binvox_for_shape_completion(
        points=observed_pc[:, 0:3], patch_size=40)
    completed_vox = binvox_rw.Voxels(cnn_voxel, partial_vox.dims,
                                     partial_vox.translate, partial_vox.scale,
                                     partial_vox.axis_order)
    # Now we save the binvox file so that it can be passed to the
    # post processing along with the partial.pcd
    temp_handle, temp_binvox_filepath = tempfile.mkstemp(
        suffix="output.binvox")
    os.close(temp_handle)
    binvox_rw.write(completed_vox, open(temp_binvox_filepath, 'w'))

    # This is the file that the post-processed mesh will be saved it.
    mesh_file = file_utils.create_file(filepath, mesh_name)
    # mesh_reconstruction tmp/completion.binvox tmp/partial.pcd tmp/post_processed.ply
    # This command will look something like

    cmd_str = "mesh_reconstruction" + " " + temp_binvox_filepath + " " + temp_pcd_filepath \
        + " " + mesh_file + " --cuda"

    subprocess.call(cmd_str.split(" "), stdout=FNULL, stderr=subprocess.STDOUT)

    # subprocess.call(cmd_str.split(" "))
    if log_pc:
        pcd_file = file_utils.create_file(filepath, pc_name)
        cmd_str = "pcl_pcd2ply -format 0" + " " + temp_pcd_filepath + " " + pcd_file
        subprocess.call(cmd_str.split(" "),
                        stdout=FNULL,
                        stderr=subprocess.STDOUT)
        map_object_to_gt(pcd_file, model_pose)

    map_object_to_gt(mesh_file, model_pose)


def save_voxel_grid(cnn_voxel_grid, file_name, folder, save_as_mat_file=False):

    if save_as_mat_file:
        voxel_file = file_utils.create_file(folder, file_name + ".mat")
        scipy.io.savemat(voxel_file, mdict={'vox': cnn_voxel_grid})
    else:
        voxel_file = file_utils.create_file(folder, file_name + ".npy")
        np.save(voxel_file, cnn_voxel_grid)


def map_object_to_gt(mesh_file, model_pose):

    mesh = plyfile.PlyData.read(mesh_file)
    mesh_vertices = np.zeros((mesh['vertex']['x'].shape[0], 3))
    mesh_vertices[:, 0] = mesh['vertex']['x']
    mesh_vertices[:, 1] = mesh['vertex']['y']
    mesh_vertices[:, 2] = mesh['vertex']['z']

    mesh_vertices = translate_and_rotate_mesh(mesh_vertices, model_pose)
    mesh['vertex']['x'] = mesh_vertices[:, 0]
    mesh['vertex']['y'] = mesh_vertices[:, 1]
    mesh['vertex']['z'] = mesh_vertices[:, 2]
    mesh.write(open(mesh_file, "wb"))


def translate_and_rotate_mesh(vertices, model_pose):

    dist_to_camera = -1
    trans_frame = PyKDL.Frame(PyKDL.Rotation.RPY(0, 0, 0),
                              PyKDL.Vector(0, 0, dist_to_camera))
    trans_matrix = tf_conversions.posemath.toMatrix(trans_frame)

    # go from camera coords to world coords
    rot_frame = PyKDL.Frame(PyKDL.Rotation.RPY(-math.pi / 2, 0, -math.pi / 2),
                            PyKDL.Vector(0, 0, 0))
    rot_matrix = tf_conversions.posemath.toMatrix(rot_frame)
    pc2 = np.ones((np.size(vertices, 0), 4))
    pc2[:, 0:3] = vertices

    # put point cloud in world frame at origin of world
    pc2_out = np.dot(trans_matrix, pc2.T)
    pc2_out = np.dot(rot_matrix, pc2_out)

    # rotate point cloud by same rotation that model went through
    pc2_out = np.dot(model_pose.T, pc2_out)
    return pc2_out.T


def voxel_grid_to_mesh(voxel_grid, filepath, name):

    v, t = mcubes.marching_cubes(voxel_grid[:, :, :], 0.5)
    unsmoothed_handle, unsmoothed_filename = tempfile.mkstemp(suffix=".dae")

    mesh_file = file_utils.create_file(filepath, name)

    mlx_script_filepath = '../filters/smooth_partial.mlx'
    cmd_str = "/home/jlundell/meshlab_source/meshlab/src/distrib/meshlabserver -i " + \
        unsmoothed_filename + " -o " + mesh_file + \
        " -s " + str(mlx_script_filepath)
    print("Calling subprocess")
    # subprocess.call(cmd_str.split())
    subprocess.call(cmd_str.split(), stdout=FNULL, stderr=subprocess.STDOUT)
    print("Finished calling subprocess")

    if os.path.exists(unsmoothed_filename):
        os.remove(unsmoothed_filename)


def log_mesh_to_file(mesh, filepath, name):
    mesh_file = file_utils.create_file(filepath, name)
    pcl.save(mesh, mesh_file)


def np_to_pcl(pc_np):
    """
    Convert nx3 numpy array to PCL pointcloud

    :type pc_np: numpy.ndarray
    :rtype pcl.PointCloud
    """

    new_pcd = pcl.PointCloud(np.array(pc_np, np.float32))
    return new_pcd


def pc_to_voxel(input_pc):
    mask = (input_pc[:, 0], input_pc[:, 1], input_pc[:, 2])
    voxel = tc.zeros((patch_size, patch_size, patch_size, 1), dtype=np.float32)
    voxel[mask] = 1
    return voxel


def calculate_mean_voxel_of_samples(samples):
    return samples.mean(axis=0)


def load_groundtruth_files(folder):
    files = file_utils.get_all_files_in_subfolder(folder)
    ground_truth_objects_file_path = file_utils.get_objects_matching_pattern(
        files, "ply")
    return ground_truth_objects_file_path


def load_shape_completed_meshes(folder, load_samples=False):
    train_models_train_views = file_utils.get_all_files_in_subfolder(
        folder + "/train_models_train_views/")
    train_models_holdout_views = file_utils.get_all_files_in_subfolder(
        folder + "/train_models_holdout_views/")
    holdout_models_holdout_views = file_utils.get_all_files_in_subfolder(
        folder + "/holdout_models_holdout_views/")
    train_models_train_views_mean_shapes = file_utils.get_objects_matching_pattern(
        train_models_train_views, "mean_shape.ply")
    train_models_holdout_views_mean_shapes = file_utils.get_objects_matching_pattern(
        train_models_holdout_views, "mean_shape.ply")
    holdout_models_holdout_views_mean_shapes = file_utils.get_objects_matching_pattern(
        holdout_models_holdout_views, "mean_shape.ply")
    if load_samples:
        train_models_train_views_sample_shapes = file_utils.get_objects_matching_pattern(
            train_models_train_views, "sample_")
        train_models_holdout_views_sample_shapes = file_utils.get_objects_matching_pattern(
            train_models_holdout_views, "sample_")
        holdout_models_holdout_views_sample_shapes = file_utils.get_objects_matching_pattern(
            holdout_models_holdout_views, "sample_")

        shape_completed_meshes = {
            "train_models_train_views":
            (train_models_train_views_mean_shapes,
             train_models_train_views_sample_shapes),
            "train_models_holdout_views":
            (train_models_holdout_views_mean_shapes,
             train_models_holdout_views_sample_shapes),
            "holdout_models_holdout_views":
            (holdout_models_holdout_views_mean_shapes,
             holdout_models_holdout_views_sample_shapes)
        }
    else:
        shape_completed_meshes = {
            "train_models_train_views": [train_models_train_views_mean_shapes],
            "train_models_holdout_views":
            [train_models_holdout_views_mean_shapes],
            "holdout_models_holdout_views":
            [holdout_models_holdout_views_mean_shapes]
        }

    return shape_completed_meshes


def load_ground_truth_and_shape_completed_meshes(
    folder_path_to_ground_truth_meshes,
    folder_path_to_completed_meshes,
    load_samples=False):
    file_path_to_all_ground_truth_meshes = load_groundtruth_files(
        folder_path_to_ground_truth_meshes)
    print(file_path_to_all_ground_truth_meshes)
    file_path_to_all_shape_completed_meshes = load_shape_completed_meshes(
        folder_path_to_completed_meshes, load_samples)
    return file_path_to_all_ground_truth_meshes, file_path_to_all_shape_completed_meshes


def get_ground_truth_mesh_corresponding_to_shape_completed_mesh(
    ground_truth_meshes, shape_completed_mesh):
    # Remove everything from the path execpt the file name, e.g. /tmp/foo/bar_poission_001_1_3_3_mean_shape.ply -> bar_poission_001_1_3_3_mean_shape.ply
    file_name = shape_completed_mesh.split("/")[-1]
    # bar_poission_001_1_3_3_mean_shape.ply-> bar_poission_001
    mesh_name = ("_").join(file_name.split("_")[:-5])

    try:
        index = [
            idx for idx, s in enumerate(ground_truth_meshes) if mesh_name in s
        ][0]
    except IndexError:
        print("Index error for mesh " + mesh_name)
    ground_truth_mesh = ground_truth_meshes[index]
    return ground_truth_mesh
