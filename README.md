# Shape Reconstruction

This repository includes code used in our work on [robust grasp planning over uncertain shape completions](https://arxiv.org/pdf/1903.00645.pdf). More specifically, it includes code to train and test our shape completion network and to do the shape reconstruction experiment.

**Authors**: Jens Lundell\
**Maintainer**: Jens Lundell, jens.lundell@aalto.fi  
**Affiliation**: Intelligent Robotics Lab, Aalto University

## Getting Started

The code was developed for python2.7 and Ubuntu 18.04.

### Dependencies

[PyTorch](https://pytorch.org/)

[Curvox](https://github.com/jsll/curvox)

[binvox-rw-py](https://github.com/dimatura/binvox-rw-py)

[python-pcl](https://github.com/strawlab/python-pcl)

[Mesh_Reconstruction](https://github.com/CRLab/Mesh_Reconstruction)

[ROS Melodic](http://wiki.ros.org/melodic) if and only if you want to integrate shape completion into, e.g., a grasping pipeline.

### Python Installation

If you are not going to run this code as a part of a ROS setup then do the following:

```
python setup.py install --user
```

### ROS Installation

If you want to do online shape completion of point-clouds acquired from a camera you need to do the following:

Clone or download the project from Github:

```
cd <PATH_TO_YOUR_CATKIN_WORKSPACE>/src
git clone git@github.com:aalto-intelligent-robotics/shape_completion_network.git
```

Compile the ROS workspace

```
cd <PATH_TO_YOUR_CATKIN_WORKSPACE>
catkin_make
```

### Download and parse training and test data

If you neither want to train a new network nor run the shape reconstruction experiment then skip this section.

To download the ground truth meshes, and the corresponding training data in the form of voxel grids either

- go to the following [link](https://drive.google.com/drive/u/1/folders/1ScywyPvZNoFzg8cn1i_gQ9OlYYe1lLg-) and download training_data.zip and ground_truth_meshes.zip
- or alternatively run the following bash script

```
cd datasets
bash download_data.sh
```

Next, to parse the data into a training and holdout set run

```
python build_dataset.py [-h] [--dataset-type [{train_test,debug}]]
                        [--path-to-dataset PATH_TO_DATASET]
                        [--dataset-seed DATASET_SEED]
                        [--use-default-split USE_DEFAULT_SPLIT]
```

If you use this training data then please also cite [the following work](#citation-for-training-data).

### Downloading pre-trained models

If you want to use one of the pre-trained models mentioned in the paper then either:

- go to the following [link](https://drive.google.com/drive/u/1/folders/1V0GmIY74MSARnRVC0ppaS1LevrS9TEbr) and download the models,
- or alternatively run the following bash script

```
cd networks
bash download_networks.sh
```

## Training and testing the network

```
shape_completion_main.py [-h]
                                [--network-model [{mc_dropout,stein,varley}]]
                                [--use-batch-norm USE_BATCH_NORM]
                                [--mode [{train,test}]] [--lr LR]
                                [--train-in-parallel TRAIN_IN_PARALLEL]
                                [--latent-dim LATENT_DIM]
                                [--num-epochs NUM_EPOCHS]
                                [--batch-size BATCH_SIZE]
                                [--num-workers NUM_WORKERS]
                                [--test-cases [{train_models_train_views,train_models_holdout_views,holdout_models_holdout_views,all}]]
                                [--num-particles NUM_PARTICLES]
                                [--num-test-samples NUM_TEST_SAMPLES]
                                [--dropout-rate DROPOUT_RATE]
                                [--checkpoint-interval CHECKPOINT_INTERVAL]
                                [--num-objects-to-test NUM_OBJECTS_TO_TEST]
                                [--save-location SAVE_LOCATION]
                                [--regularization REGULARIZATION]
                                [--use-cuda USE_CUDA]
                                [--net-recover NET_RECOVER]
                                [--net-recover-name NET_RECOVER_NAME]
                                [--net-recover-epoch NET_RECOVER_EPOCH]
                                [--use-skip-connections USE_SKIP_CONNECTIONS]
                                [--debug DEBUG] [--save-samples SAVE_SAMPLES]
                                [--save-mesh SAVE_MESH]
                                [--save-voxel-grid SAVE_VOXEL_GRID]
```

### Training a new network

```
shape_completion_main.py  --mode train [-h]
                                [--network-model [{mc_dropout,stein,varley}]]
                                [--use-batch-norm USE_BATCH_NORM]
                                [--lr LR]
                                [--train-in-parallel TRAIN_IN_PARALLEL]
                                [--latent-dim LATENT_DIM]
                                [--num-epochs NUM_EPOCHS]
                                [--batch-size BATCH_SIZE]
                                [--num-workers NUM_WORKERS]
                                [--num-particles NUM_PARTICLES]
                                [--dropout-rate DROPOUT_RATE]
                                [--checkpoint-interval CHECKPOINT_INTERVAL]
                                [--save-location SAVE_LOCATION]
                                [--regularization REGULARIZATION]
                                [--use-cuda USE_CUDA]
                                [--net-recover NET_RECOVER]
                                [--net-recover-name NET_RECOVER_NAME]
                                [--net-recover-epoch NET_RECOVER_EPOCH]
                                [--use-skip-connections USE_SKIP_CONNECTIONS]
                                [--debug DEBUG]
```

### Testing a network

When testing the network, the network first reads in point-clouds from the test-set, converts these into voxel grids, shape completes the voxel grid, ,, combines the original point-cloud with the shape completed voxel grid, and finally runs marching cube algorithm to get a mesh.

For testing a network run

```
shape_completion_main.py --mode test --batch-size 1 --num-workers 1
                            [-h]
                                [--network-model [{mc_dropout,stein,varley}]]
                                [--use-batch-norm USE_BATCH_NORM]
                                [--latent-dim LATENT_DIM]
                                [--test-cases [{train_models_train_views,train_models_holdout_views,holdout_models_holdout_views,all}]]
                                [--num-test-samples NUM_TEST_SAMPLES]
                                [--dropout-rate DROPOUT_RATE]
                                [--num-objects-to-test NUM_OBJECTS_TO_TEST]
                                [--save-location SAVE_LOCATION]
                                [--use-cuda USE_CUDA]
                                [--net-recover NET_RECOVER]
                                [--net-recover-name NET_RECOVER_NAME]
                                [--use-skip-connections USE_SKIP_CONNECTIONS]
                                [--save-samples SAVE_SAMPLES]
                                [--save-mesh SAVE_MESH]
                                [--save-voxel-grid SAVE_VOshape completed meshes b
```

## Shape reconstruction

In this section, we detail how you can run the shape reconstruction experiment as explained in Section IV-B in our [paper](https://arxiv.org/pdf/1903.00645.pdf). However, before you can run the experiment you need some shape completed meshes. You can either generate new ones by following [this](testing-a-network) or, alternatively, download some already shape completed meshes [here](https://drive.google.com/drive/folders/128kbeCBe3W3leGJcV3fmtLMu35jHSPuD?usp=sharing).

After you have procured the meshes, run

```

shape_reconstruction_main.py [-h]
                                    [--completion-method [{mc_dropout,stein,varley}]]
                                    [--save-dir SAVE_DIR]
                                    [--ground-truth-data-dir GROUND_TRUTH_DATA_DIR]
                                    [--shape-completed-data-dir SHAPE_COMPLETED_DATA_DIR]

```

# Citation

This repository contains code corresponding to our work on [robust grasp planning over uncertain shape completions](https://arxiv.org/pdf/1903.00645.pdf). If you use the code for your research, please cite

```

@article{lundell2019robust,
  title={Robust Grasp Planning Over Uncertain Shape Completions},
  author={Lundell, Jens and Verdoja, Francesco and Kyrki, Ville},
  journal={arXiv preprint arXiv:1903.00645},
  year={2019}
}

```

## Citation for training data

The original training data was not created by us but by Jacob Varley et al. and we only have the authors approval to redistribute. Thus, if you use the training data in your research project then please also cite the original work

```

@inproceedings{varley2017shape,
  title={Shape completion enabled robotic grasping},
  author={Varley, Jacob and DeChant, Chad and Richardson, Adam and Ruales, Joaqu{\'\i}n and Allen, Peter},
  booktitle={2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2017},
  organization={IEEE}
}

```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
