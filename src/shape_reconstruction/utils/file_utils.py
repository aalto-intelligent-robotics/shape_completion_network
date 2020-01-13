import os
import shutil

import numpy as np
import yaml


def write_to_file(file, data):
    file.write(data)


def remove_all_elements_in_list_mathching_a_string(list, string):
    return [x for x in list if string not in x]


def get_all_files_in_folder(folder):
    files = []
    files = os.listdir(folder)
    return [folder + "/" + x for x in files]


def get_all_subfolders(folder):
    return os.listdir(folder)


def get_all_files_in_subfolder(folder):
    allFiles = []
    subdirs = [x[0] for x in os.walk(folder)]
    subdirs.pop(0)
    for subdir in subdirs:
        files = get_all_files_in_folder(subdir)
        allFiles += files
    return allFiles


def get_objects_matching_pattern(objects, pattern):
    matched_objects = []
    for obj in objects:
        if pattern in obj:
            matched_objects.append(obj)
    return matched_objects


def strip_file_definition(file_name):
    file_name = file_name.split(".")[-2]
    return file_name.split("/")[-1]


def log_data_to_file(data, log_file):
    with open(log_file, "a") as data_file:
        data_file.write(str(data) + '\n')


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def create_file(folder, filename=""):
    create_folder(folder)
    try:
        f = open(folder + filename, "w")
        f.close()
    except IOError:
        print("Could not open file " + folder + filename)
    return folder + filename


def save_numpy_arr_to_file(numpyArray, folder, filename):
    resultFile = create_file(folder, filename)
    np.save(resultFile, numpyArray)


def remove_folders(folders):
    for folder in folders:
        remove_folder(folder)


def remove_folder(folder):
    try:
        shutil.rmtree(folder)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def file_exist(folder, candidateObject):
    filePath = folder + candidateObject
    return os.path.isfile(filePath)


def read_yaml_file(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def get_file_name_and_path_from_folder_path(folder_path):
    temp = folder_path.split("/")
    file_name = temp.pop()
    folder_path = ("/").join(temp) + "/"
    return file_name, folder_path
