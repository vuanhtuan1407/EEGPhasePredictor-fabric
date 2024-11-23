import torch
from torch import nn

from ... import params
from ...utils.config_utils import load_yaml_config
from ...utils.data_utils import LABEL_DICT


class MNAPooling1D(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.max_pooling = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.average_pooling = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        mx = self.max_pooling(x)
        avg = self.average_pooling(x)
        return torch.concat([mx, avg], dim=1)


class BiMaxPooling1D(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.max_pooling = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        mx1 = self.max_pooling(x)
        mx2 = self.max_pooling(-x)
        return torch.concat([mx1, mx2], dim=1)


class Conv1DLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            pooling_kernel_size,
            pooling_stride,
            pooling_padding=0,
            dropout=0.1
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride, padding=pooling_padding)
        )

    def forward(self, x):
        return self.conv(x)


class CNN1DModel(nn.Module):
    def __init__(self, yml_config_file='cnn1d_config.yml'):
        super().__init__()
        self.type = 'cnn1d'
        self.config = load_yaml_config(yml_config_file)
        conv_layer_config = self.config['conv_layers']
        self.conv1d = nn.Sequential()
        conv_in_channels = 3
        for i, layer_config in enumerate(conv_layer_config):
            conv_layer = Conv1DLayer(
                in_channels=conv_in_channels,
                out_channels=layer_config['out_channels'],
                kernel_size=layer_config['kernel_size'],
                stride=layer_config['stride'],
                padding=layer_config['padding'],
                pooling_kernel_size=layer_config['pooling_kernel_size'],
                pooling_stride=layer_config['pooling_stride'],
                pooling_padding=layer_config['pooling_padding'],
                dropout=layer_config['dropout']
            )
            conv_in_channels = layer_config['out_channels']
            self.conv1d.append(conv_layer)

        self.flatten = nn.Flatten()

        ff_in_features = 3834
        self.ff1 = nn.Sequential(
            nn.Linear(in_features=ff_in_features, out_features=ff_in_features * 2),
            nn.ReLU(),
        )
        self.ff2 = nn.Sequential(
            nn.Linear(in_features=ff_in_features * 2, out_features=len(LABEL_DICT) * params.W_OUT),
            nn.ReLU(),
        )
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.flatten(x)
        x = self.ff1(x)
        x = self.ff2(x)
        x = x.reshape(-1, len(LABEL_DICT), params.W_OUT)
        x = x.transpose(1, 2)
        return x
