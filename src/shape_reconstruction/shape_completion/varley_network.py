import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()


def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


class KitModel(nn.Module):
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.conv_3D_1 = self.__conv(3,
                                     name='conv_3D_1',
                                     in_channels=1,
                                     out_channels=64,
                                     kernel_size=(4L, 4L, 4L),
                                     stride=(1L, 1L, 1L),
                                     groups=1,
                                     bias=True)
        self.conv_3D_2 = self.__conv(3,
                                     name='conv_3D_2',
                                     in_channels=64,
                                     out_channels=64,
                                     kernel_size=(4L, 4L, 4L),
                                     stride=(1L, 1L, 1L),
                                     groups=1,
                                     bias=True)
        self.conv_3D_3 = self.__conv(3,
                                     name='conv_3D_3',
                                     in_channels=64,
                                     out_channels=64,
                                     kernel_size=(4L, 4L, 4L),
                                     stride=(1L, 1L, 1L),
                                     groups=1,
                                     bias=True)
        self.dense_compress_1 = self.__dense(name='dense_compress_1',
                                             in_features=4096,
                                             out_features=10976,
                                             bias=True)
        self.dense_compress_2 = self.__dense(name='dense_compress_2',
                                             in_features=10976,
                                             out_features=1372,
                                             bias=True)
        self.dense_reconstruct = self.__dense(name='dense_reconstruct',
                                              in_features=1372,
                                              out_features=64000,
                                              bias=True)

    def forward(self, x):
        conv_3D_1 = self.conv_3D_1(x)
        conv_3D_1_activation = F.relu(conv_3D_1)
        max_pool_1 = F.max_pool3d(conv_3D_1_activation,
                                  kernel_size=(2L, 2L, 2L),
                                  stride=(2L, 2L, 2L),
                                  padding=0,
                                  ceil_mode=False)
        conv_3D_2 = self.conv_3D_2(max_pool_1)
        conv_3D_2_activation = F.relu(conv_3D_2)
        max_pool_2 = F.max_pool3d(conv_3D_2_activation,
                                  kernel_size=(2L, 2L, 2L),
                                  stride=(2L, 2L, 2L),
                                  padding=0,
                                  ceil_mode=False)
        conv_3D_3 = self.conv_3D_3(max_pool_2)
        conv_3D_3_activation = F.relu(conv_3D_3)
        flatten = conv_3D_3_activation.view(conv_3D_3_activation.size(0), -1)
        dense_compress_1 = self.dense_compress_1(flatten)
        dense_compress_1_activation = F.relu(dense_compress_1)
        dense_compress_2 = self.dense_compress_2(dense_compress_1_activation)
        dense_compress_2_activation = F.relu(dense_compress_2)
        dense_reconstruct = self.dense_reconstruct(dense_compress_2_activation)
        dense_reconstruct_activation = torch.sigmoid(dense_reconstruct)
        return dense_reconstruct_activation

    def set_device(self, device):
        pass

    def test(self, x, y, num_samples=0):
        return 0, self.forward(x).squeeze().reshape(40, 40,
                                                    40).cpu().detach().numpy()

    @staticmethod
    def __conv(dim, name, **kwargs):
        if dim == 1: layer = nn.Conv1d(**kwargs)
        elif dim == 2: layer = nn.Conv2d(**kwargs)
        elif dim == 3: layer = nn.Conv3d(**kwargs)
        else: raise NotImplementedError()

        layer.state_dict()['weight'].copy_(
            torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(
                torch.from_numpy(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(
            torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(
                torch.from_numpy(__weights_dict[name]['bias']))
        return layer
