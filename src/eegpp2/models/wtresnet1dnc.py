import numpy as np
import pywt
import ptwt
import torch
from torch import nn

from .. import params
from ..models.baseline.resnet import ResNet50
from ..utils.config_utils import load_yaml_config
from ..utils.data_utils import LABEL_DICT


class WTEmbedding(nn.Module):
    def __init__(self, emb_size, wavelet='morl'):
        super().__init__()
        self.scales = np.geomspace(1, 1e3, num=emb_size)
        self.wavelet = wavelet
        time = np.linspace(0.0, 4.0 * params.W_OUT, num=1024 * params.W_OUT)
        self.sample_period = np.diff(time).mean()

    def forward2(self, x):
        device = x.device
        x = x.cpu().numpy()
        x, _ = pywt.cwt(x, self.scales, self.wavelet, self.sample_period, axis=-1)
        x = torch.from_numpy(x).float().to(device)
        x = torch.abs(x)
        x = torch.transpose(x, 0, 1)
        return x

    def forward(self, x):
        x, _ = ptwt.cwt(x, self.scales, self.wavelet, self.sample_period)
        x = torch.abs(x)
        x = torch.transpose(x, 0, 1)
        x = x.float()
        return x


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


class WTResnet501DnCModel(nn.Module):
    def __init__(self, yml_config_file='wtresnet501dnc_config.yml'):
        super().__init__()
        self.type = 'wtresnet501dnc'
        self.config = load_yaml_config(yml_config_file)
        num_chains = self.config['num_chains']
        self.chains = nn.ModuleList([nn.Sequential() for _ in range(num_chains)])
        out_dim = 512
        for chain in self.chains:
            input_embedding = WTEmbedding(
                emb_size=self.config['emb_size'],
                wavelet=self.config['wavelet'],
            )
            resnet501d = ResNet50(in_channels=self.config['emb_size'], num_classes=out_dim * 5)
            fc = FC(in_dim=out_dim * 5, out_dim=out_dim)
            chain.add_module('input_embedding', input_embedding)
            chain.add_module('resnet501d', resnet501d)
            chain.add_module('fc', fc)

        self.classifier = FC(in_dim=out_dim * num_chains, out_dim=params.W_OUT * len(LABEL_DICT))
        self.classifier_binary = FC(in_dim=out_dim * num_chains + params.W_OUT * len(LABEL_DICT),
                                    out_dim=params.W_OUT * 2)

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
