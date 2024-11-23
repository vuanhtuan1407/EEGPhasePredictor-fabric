import torch
from torch import nn

from .. import params
from ..utils.config_utils import load_yaml_config
from ..utils.data_utils import LABEL_DICT


class MNAPooling1D(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.max_pooling = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.average_pooling = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        mx = self.max_pooling(x)
        avg = self.average_pooling(x)
        return torch.concat([mx, avg], dim=1)


class BiMaxPooling1D(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.max_pooling = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        mx1 = self.max_pooling(x)
        mx2 = self.max_pooling(-x)
        return torch.concat([mx1, mx2], dim=1)


class FFTEmbedding(nn.Module):
    def __init__(self, n_fft, norm='forward', tiled=False, d_model=None):
        super().__init__()
        self.n_fft = n_fft
        self.norm = norm
        self.tiled = tiled
        self.d_model = d_model

    def forward(self, x):
        x = torch.fft.rfft(x, n=self.n_fft, norm=self.norm)
        x = torch.abs(x)
        if self.tiled and isinstance(self.d_model, int):
            x = torch.tile(x, dims=(self.d_model,))
            x = torch.reshape(x, (x.size(0), self.n_fft // 2 + 1, self.d_model))
        else:
            x = torch.unsqueeze(x, dim=-1)
            x = torch.transpose(x, 1, 2)
        return x


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


class FC(nn.Module):
    def __init__(
            self,
            in_dim: int = 1024,
            dim_feedforward: int = 2048,
            out_dim: int = 7,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, dim_feedforward),
            nn.ReLU(),
        )
        self.ff2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.ff1(x)
        x = self.ff2(x)
        return x


class FFTCNN1DnCModel(nn.Module):
    def __init__(self, yml_config_file='fftcnn1dnc_config.yml'):
        super().__init__()
        self.type = 'fftcnn1dnc'
        self.config = load_yaml_config(yml_config_file)
        num_chains = self.config['num_chains']
        self.chains = nn.ModuleList([nn.Sequential() for _ in range(num_chains)])
        out_dim = 512
        for chain in self.chains:
            input_embedding = FFTEmbedding(
                n_fft=self.config['n_fft']
            )
            chain.add_module('input_embedding', input_embedding)
            in_channels = 1
            for i, layer_config in enumerate(self.config['conv_layers']):
                conv_layer = Conv1DLayer(
                    in_channels=in_channels,
                    out_channels=layer_config['out_channels'],
                    kernel_size=layer_config['kernel_size'],
                    stride=layer_config['stride'],
                    padding=layer_config['padding'],
                    pooling_kernel_size=layer_config['pooling_kernel_size'],
                    pooling_stride=layer_config['pooling_stride'],
                    pooling_padding=layer_config['pooling_padding'],
                    dropout=layer_config['dropout']
                )
                chain.add_module(name=layer_config['name'], module=conv_layer)
                in_channels = layer_config['out_channels']
            flatten = nn.Flatten()
            fc = FC(in_dim=1024 * 15, out_dim=out_dim)
            chain.add_module('flatten', flatten)
            chain.add_module('fc', fc)

        self.classifier = FC(in_dim=out_dim * num_chains, out_dim=params.W_OUT * len(LABEL_DICT))
        self.classifier_binary = FC(in_dim=out_dim * num_chains + params.W_OUT * len(LABEL_DICT), out_dim=params.W_OUT * 2)

    def forward(self, x):
        xis = []
        for i, chain in enumerate(self.chains):
            xi = x[:, i, :]
            xi = chain(xi)
            xis.append(xi)

        out = torch.concat(xis, dim=-1)
        outx = out
        out = self.classifier(out)
        out = out.reshape(out.size(0), params.W_OUT, -1)
        out2x = torch.concat([outx, out.reshape(out.size(0), -1)], dim=1)
        out2 = self.classifier_binary(out2x)
        out2 = out2.reshape(out2.size(0), params.W_OUT, -1)
        return out, out2

    @staticmethod
    def chain_forward(chain, x):
        return chain(x)
