from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# filename = f'{params.MODEL_TYPE}.pkl'
# model_checkpoint = ModelCheckpoint(
#     dirpath=str(Path(OUT_DIR, 'checkpoints')),
#     filename=filename,
#     enable_version_counter=False,
#     monitor='val_loss',
#     every_n_epochs=1,
#     save_on_train_epoch_end=False,
#     mode='min',
#     save_top_k=1,
# )
# model_checkpoint.FILE_EXTENSION = '.pkl'

model_checkpoint = ModelCheckpoint(monitor='val_loss', mode='min')
model_checkpoint.FILE_EXTENSION = '.pkl'

early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=11,
    verbose=True,
    check_finite=True,
    mode="min"
)
