from src.eegpp.models.baseline.cnn_model import CNN1DModel


def get_model(model_type):
    if model_type == 'cnn1d':
        return CNN1DModel()
    elif model_type == 'cnn1d2c':
        return None
    else:
        raise ValueError(f'Model type {model_type} not supported')


if __name__ == '__main__':
    pass
