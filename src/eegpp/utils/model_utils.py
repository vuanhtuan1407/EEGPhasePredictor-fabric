from src.eegpp.models.baseline.cnn1d import CNN1DModel
from src.eegpp.models.baseline.transformer import TransformerModel
from src.eegpp.models.cnn1d2c import CNN1D2CModel
from src.eegpp.models.fft2c import FFT2CModel
from src.eegpp.models.stftcnn1dnc import STFTCNN1DnCModel
from src.eegpp.models.stfttransnc import STFTTransnCModel


def get_model(model_type, yml_config_file=None):
    if model_type == 'cnn1d':
        return CNN1DModel()
    elif model_type == 'transformer':
        return TransformerModel()
    elif model_type == 'cnn1d2c':
        return CNN1D2CModel()
    elif model_type == 'fft2c':
        return FFT2CModel()
    elif model_type == 'stfttransnc':
        return STFTTransnCModel()
    elif model_type == 'stftcnn1dnc':
        return STFTCNN1DnCModel()
    else:
        raise ValueError(f'Model type {model_type} not supported')


def freeze_parameters(model):
    # use with torch.no_grad()
    for param in model.parameters():
        param.requires_grad = False
    # return model


def unfreeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = True
    # return model


if __name__ == '__main__':
    pass
