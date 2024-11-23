import math

import torch
from torch import nn

from .. import params
from ..utils.config_utils import load_yaml_config
from ..utils.data_utils import LABEL_DICT


class STFTEmbedding(nn.Module):
    def __init__(
            self,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            return_complex=True,
            normalized=True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.return_complex = return_complex
        self.normalized = normalized

    def forward(self, x):
        window = torch.hamming_window(self.win_length, device=x.device)
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=window,
                       normalized=self.normalized, return_complex=self.return_complex, onesided=False)
        x = torch.sqrt(x.real ** 2 + x.imag ** 2)
        # x = x[:, 1:, :]  # remove first/last frequency
        x = torch.transpose(x, 1, 2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, e_dim: int = 512, dropout: float = 0.1, max_len: int = 50000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, e_dim, 2) * (-math.log(10000.0) / e_dim))
        pe = torch.zeros(max_len, e_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe_x = self.pe[:, :x.size(1), :]
        x = x + pe_x
        x = self.dropout(x)
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


class STFTTransnCModel(nn.Module):
    def __init__(self, yml_config_file='stfttransnc_config.yml'):
        super().__init__()
        self.type = 'stfttransnc'
        self.config = load_yaml_config(yml_config_file)
        num_chains = self.config['num_chains']
        d_model = self.config['n_fft']
        out_dim = 512
        self.chains = nn.ModuleList([nn.Sequential() for _ in range(num_chains)])
        for chain in self.chains:
            input_embedding = STFTEmbedding(
                n_fft=self.config['n_fft'],
                win_length=self.config['win_length'],
                hop_length=self.config['hop_length'],
                return_complex=self.config['return_complex'],
                normalized=self.config['normalized']
            )
            positional_encoding = PositionalEncoding(
                e_dim=d_model,
                dropout=self.config['dropout'],
            )
            transformer_encoder = TransformerEncoderBlock(
                d_model=d_model,
                nhead=self.config['nhead'],
                dim_feedforward=self.config['dim_feedforward'],
                dropout=self.config['dropout'],
                num_layers=self.config['num_layers']
            )
            fc = FC(in_dim=d_model, out_dim=out_dim)
            chain.add_module('input_embedding', input_embedding)
            chain.add_module('positional_encoding', positional_encoding)
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


if __name__ == "__main__":
    t = torch.randn(2, 3, 4)
    cls = torch.randn(2, 1, 4)
    rs = torch.concat([cls, t], dim=1)
    print(rs.shape)
