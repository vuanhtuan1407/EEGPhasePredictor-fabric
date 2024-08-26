from src.eegpp.models.baseline.cnn1d import CNN1DModel
from src.eegpp.models.fft_2c import FFT2CModel


def get_model(model_type):
    if model_type == 'cnn1d':
        return CNN1DModel()
    elif model_type == 'cnn1d2c':
        return None
    elif model_type == 'fft2c':
        return FFT2CModel()
    else:
        raise ValueError(f'Model type {model_type} not supported')


if __name__ == '__main__':
    pass
