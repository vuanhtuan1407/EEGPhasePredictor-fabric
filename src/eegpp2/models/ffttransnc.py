import torch.fft
from torch import nn

from .. import params
from ..utils.config_utils import load_yaml_config
from ..utils.data_utils import LABEL_DICT


class FFTEmbedding(nn.Module):
    def __init__(self, n_fft, d_model=1, norm='forward'):
        super().__init__()
        self.n_fft = n_fft
        self.d_model = d_model
        self.norm = norm

    def forward(self, x):
        x = torch.fft.rfft(x, n=self.n_fft, norm=self.norm)
        x = torch.abs(x)
        x = torch.tile(x, dims=(self.d_model,))
        x = torch.reshape(x, (x.size(0), self.n_fft // 2 + 1, self.d_model))
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            nhead: int = 8,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            num_layers: int = 2,
            auto_add_cls=True
    ):
        super().__init__()
        self.auto_add_cls = auto_add_cls
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        if self.auto_add_cls:
            cls = torch.zeros((x.size(0), 1, x.size(2)), device=x.device)
            x = torch.concat([cls, x], dim=1)
        x = self.encoder(x)
        x = x[:, 0, :]
        # x = torch.mean(x, dim=1)
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


class FFTTransnCModel(nn.Module):
    def __init__(self, yml_config_file='ffttransnc_config.yml'):
        super().__init__()
        self.type = 'ffttransnc'
        self.config = load_yaml_config(yml_config_file)
        num_chains = self.config['num_chains']
        out_dim = 512
        self.chains = nn.ModuleList([nn.Sequential() for _ in range(num_chains)])
        for chain in self.chains:
            input_embedding = FFTEmbedding(
                n_fft=self.config['n_fft'],
                d_model=self.config['d_model'],
            )
            transformer_encoder = TransformerEncoderBlock(
                d_model=self.config['d_model'],
                nhead=self.config['nhead'],
                dim_feedforward=self.config['dim_feedforward'],
                dropout=self.config['dropout'],
                num_layers=self.config['num_layers']
            )
            fc = FC(in_dim=self.config['d_model'], out_dim=out_dim)
            chain.add_module('input_embedding', input_embedding)
            chain.add_module('transformer_encoder', transformer_encoder)
            chain.add_module('fc', fc)

        self.classifier = FC(
            in_dim=out_dim * num_chains,
            out_dim=params.W_OUT * len(LABEL_DICT),
        )
        self.classifier_binary = FC(
            in_dim=out_dim * num_chains + params.W_OUT * len(LABEL_DICT),
            out_dim=params.W_OUT * 2,
        )

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
        for k, layer in chain.items():
            x = layer(x)
        return x
