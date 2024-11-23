import math

import torch
from torch import nn

# from ...utils.model_utils import freeze_parameters
from ... import params
from ...utils.config_utils import load_yaml_config
from ...utils.data_utils import LABEL_DICT


class InputEmbedding(nn.Module):
    def __init__(
            self,
            inp_dim=3,
            e_dim=1024,
    ):
        super().__init__()
        self.inp_dim = inp_dim
        self.e_dim = e_dim
        self.emb_matrix = None

    def forward2(self, x):
        if x.size(1) != self.inp_dim:
            raise ValueError("Wrong input shape in Transformer Embedding!")
        x = torch.transpose(x, 1, 2)
        return x

    def forward(self, x):
        if x.size(1) != self.inp_dim:
            raise ValueError("Wrong input shape in Transformer Embedding!")
        x = torch.transpose(x, 1, 2)
        self.emb_matrix = nn.Parameter(torch.ones((x.size(0), self.inp_dim, self.e_dim), device=x.device))
        return torch.matmul(x, self.emb_matrix)


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
            num_layers: int = 2
    ):
        super().__init__()
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
        x = self.encoder(x)

        # Use [CLS]
        # x = x[:, 0, :]
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            nhead: int = 8,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            num_layers: int = 2
    ):
        super().__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_layers
        )

    def forward(self, obj_queries, x):
        x = self.decoder(obj_queries, x)
        return x


class Classifier(nn.Module):
    def __init__(
            self,
            in_features: int = 1024,
            dim_feedforward: int = 2048,
            out_features: int = 7,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, dim_feedforward),
            nn.ReLU(),
        )
        self.ff2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.ff1(x)
        x = self.ff2(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, yml_config_file='transformer_config.yml'):
        super().__init__()
        self.type = 'transformer'
        self.config = load_yaml_config(yml_config_file)
        self.input_embedding = InputEmbedding(
            e_dim=self.config['d_model']
        )
        self.positional_encoding = PositionalEncoding(
            e_dim=self.config['d_model'],
            dropout=self.config['dropout']
        )
        self.transformer_encoder = TransformerEncoderBlock(
            d_model=self.config['d_model'],
            nhead=self.config['nhead_encoder'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=self.config['dropout'],
            num_layers=self.config['num_encoder_layers']
        )
        self.transformer_decoder = TransformerDecoderBlock(
            d_model=self.config['d_model'],
            nhead=self.config['nhead_decoder'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=self.config['dropout'],
            num_layers=self.config['num_decoder_layers']
        )
        self.classifier = Classifier(
            in_features=self.config['d_model'],
            out_features=len(LABEL_DICT) * params.W_OUT,
        )
        self.classifier_binary = Classifier(
            in_features=self.config['d_model'] + len(LABEL_DICT) * params.W_OUT,
            out_features=2 * params.W_OUT,
        )

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        # out = x
        obj_queries = torch.zeros(x.size(0), params.NUM_QUERIES, self.config['d_model'], device=x.device)
        out = self.transformer_decoder(obj_queries, x)
        out = torch.mean(out, dim=1)
        outx = out
        out = self.classifier(out)
        out = out.reshape(out.size(0), params.W_OUT, -1)
        tmpx = torch.concat([outx, out.reshape(out.size(0), -1)], dim=1)
        out2 = self.classifier_binary(tmpx)
        out2 = out2.reshape(out2.size(0), params.W_OUT, -1)
        return out, out2


if __name__ == '__main__':
    model = TransformerModel()
    inp = torch.randn((1, 3, 4))
    out, out2 = model(inp)
    print(out.shape, out2.shape)
    # freeze_parameters(model)
