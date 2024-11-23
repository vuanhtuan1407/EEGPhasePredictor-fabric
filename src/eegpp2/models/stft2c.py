from torch import nn

from ..utils.config_utils import load_yaml_config


class STFT2CModel(nn.Module):
    def __int__(self, yml_config_file='stft2c_config.yml'):
        super().__int__()
        self.type = 'stft2c'
        self.config = load_yaml_config(yml_config_file)

    def forward(self, x):
        pass
