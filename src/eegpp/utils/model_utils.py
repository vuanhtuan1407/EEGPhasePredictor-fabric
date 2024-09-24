import torch
from torchinfo import summary

from src.eegpp import params
from src.eegpp.models.baseline.cnn1d import CNN1DModel
from src.eegpp.models.baseline.transformer import TransformerModel
from src.eegpp.models.cnn1d2c import CNN1D2CModel
from src.eegpp.models.fft2c import FFT2CModel
from src.eegpp.models.ffttransnc import FFTTransnCModel
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
    elif model_type == 'ffttransnc':
        return FFTTransnCModel()
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


# Check if model use Fourier Transform in first signal extraction
def check_using_ft(model_type):
    ft_methods = ['ft', 'fft', 'stft']
    return any(method in model_type for method in ft_methods)


def summarize_model(model_type, input_size, verbose=0):
    input_data = torch.zeros(input_size)
    model_summary = summary(model=get_model(model_type), input_data=input_data, verbose=verbose)
    model_summary = str(model_summary)
    return model_summary


if __name__ == '__main__':
    model_type = 'ffttransnc'
    input_size = (10, 3, 1 * params.MAX_SEQ_SIZE)
    model_summary = summarize_model(model_type=model_type, input_size=input_size, verbose=1)
