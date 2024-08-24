from src.eegpp import params
from src.eegpp.trainer import EEGKFoldTrainer
from src.eegpp.utils.model_utils import get_model

if __name__ == "__main__":
    model = get_model(params.MODEL_TYPE)
    trainer = EEGKFoldTrainer(
        model=model,
        lr=params.LEARNING_RATE,
        n_epochs=params.NUM_EPOCHS,
        n_splits=params.N_SPLITS,
        accelerator=params.ACCELERATOR,
        devices=params.DEVICES,
    )
    trainer.fit()
