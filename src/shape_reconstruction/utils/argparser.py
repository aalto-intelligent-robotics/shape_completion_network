import argparse
import os


def checkPositive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value" % value)
    return ivalue


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_test_parser():
    parser = argparse.ArgumentParser(
        description='Train the shape completion network.')
    parser.add_argument('--network-model',
                        type=str,
                        choices=["mc_dropout", "stein", "varley"],
                        default="mc_dropout",
                        nargs='?',
                        help='What network type to use')
    parser.add_argument(
        '--use-batch-norm',
        type=str2bool,
        default=True,
        help='Whether to add batch normalization to the networks')
    parser.add_argument('--mode',
                        type=str,
                        choices=["train", "test"],
                        default="train",
                        nargs='?',
                        help='Whether we want to train or test a network')
    parser.add_argument('--lr',
                        default=1.0e-3,
                        type=float,
                        help='Learning rate')
    parser.add_argument(
        '--train-in-parallel',
        default=False,
        type=str2bool,
        help='Whether to train the model in parallel on multiple GPUs')

    parser.add_argument('--latent-dim',
                        default=2000,
                        type=int,
                        help='Number of latent dimensions')
    parser.add_argument('--num-epochs',
                        default=10,
                        type=int,
                        help='Number of epochs')
    parser.add_argument('--batch-size',
                        default=256,
                        type=int,
                        help='Batch size')
    parser.add_argument('--num-workers',
                        default=8,
                        type=int,
                        help='Num processes utilized')
    parser.add_argument('--test-cases',
                        type=str,
                        choices=[
                            "train_models_train_views",
                            "train_models_holdout_views",
                            "holdout_models_holdout_views", "all"
                        ],
                        default="train_models_train_views",
                        nargs="?",
                        help='The test cases')
    parser.add_argument(
        '--num-particles',
        default=10,
        type=int,
        help=
        'Number of particles (samples) to evaluate the kernel when training the Stein network.'
    )
    parser.add_argument(
        '--num-test-samples',
        default=10,
        type=int,
        help=
        'Number of samples to generate at test time for each input for either Stein or mc dropout network.'
    )
    parser.add_argument('--dropout-rate',
                        default=0.2,
                        type=float,
                        help='Dropout rate')
    parser.add_argument(
        '--checkpoint-interval',
        default=10,
        type=int,
        help='How many epochs before saving a model checkpoint')
    parser.add_argument('--num-objects-to-test',
                        default=-1,
                        type=int,
                        help='How many objects in each test set to test')
    parser.add_argument('--save-location',
                        default='../networks/training_results/',
                        type=str,
                        help='Results are saved to a given folder name')
    parser.add_argument('--regularization',
                        type=checkPositive,
                        default=1,
                        help='How much\
                        regularization to add for training stein networks')
    parser.add_argument('--use-cuda', type=str2bool, default=True)
    parser.add_argument('--net-recover',
                        type=str2bool,
                        default=False,
                        help='If we want to train from a previous instant\
                        ')
    parser.add_argument(
        '--net-recover-name',
        type=str,
        default=os.path.abspath(
            'networks/pre_trained_networks/mc_dropout_rate_0.2_lat_2000.model'
        ),
        help='Path to the network to recover')
    parser.add_argument('--net-recover-epoch',
                        type=checkPositive,
                        default=0,
                        help='Epoch we recover from')
    parser.add_argument('--use-skip-connections',
                        type=str2bool,
                        default=False,
                        help='Turn skip-connections on or off')
    parser.add_argument('--debug',
                        type=str2bool,
                        default=False,
                        help='Turn debug mode on')
    parser.add_argument(
        '--save-samples',
        type=str2bool,
        default=False,
        help=
        'Indicate if we at test time want to save the shape completed samples')
    parser.add_argument(
        '--save-mesh',
        type=str2bool,
        default=False,
        help=
        'Indicate if we at test time want to convert and save voxel grids as meshes'
    )
    parser.add_argument(
        '--save-voxel-grid',
        type=str2bool,
        default=False,
        help=
        'Indicate if we at test time want to save the shape completed voxel grids'
    )
    return parser


def reconstruction_parser():
    parser = argparse.ArgumentParser(
        description='Do the reconstruction experiment.')
    parser.add_argument(
        '--completion-method',
        type=str,
        choices=["mc_dropout", "stein", "varley"],
        default="mc_dropout",
        nargs='?',
        help='What completion method was used to generate the data')
    parser.add_argument('--save-dir',
                        default='/tmp/',
                        type=str,
                        help='Specify the save directory')
    parser.add_argument(
        '--ground-truth-data-dir',
        default=os.path.abspath('datasets/ground_truth_meshes/'),
        type=str,
        help='Specify the directory containing the ground truth data')
    parser.add_argument(
        '--shape-completed-data-dir',
        default=os.path.abspath('datasets/shape_completed_data/'),
        type=str,
        help='Specify the directory containing the shape completed data')
    return parser