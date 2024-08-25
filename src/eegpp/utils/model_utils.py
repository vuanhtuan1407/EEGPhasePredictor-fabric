from src.eegpp.models.baseline.cnn1d import CNN1DModel
from src.eegpp.models.fft_2c_Wout import FFT2CWOutModel


def get_model(model_type):
    if model_type == 'cnn1d':
        return CNN1DModel()
    elif model_type == 'cnn1d2c':
        return None
    elif model_type == 'fft2c':
        return FFT2CWOutModel()
    else:
        raise ValueError(f'Model type {model_type} not supported')


if __name__ == '__main__':
    pass
