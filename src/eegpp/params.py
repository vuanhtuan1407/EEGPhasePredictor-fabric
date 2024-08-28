BATCH_SIZE = 8
ENABLE_LOGGING = False
ENABLE_CHECKPOINTING = True
DEVICES = 'auto'
ACCELERATOR = 'auto'
NUM_EPOCHS = 10
MAX_SEQ_SIZE = 1024
NUM_CLASSES = 7  # 6 standard + 1 other
MODEL_TYPE = 'fft2c'
RESUME_CKPT = False
W_OUT = 5
NUM_WORKERS = 0
DATASET_FILE_IDX = 'all'
LEARNING_RATE = 1e-3
RD_SEED = 42
N_SPLITS = 5
POS_IDX = int(W_OUT / 2)  # position index of the main segment
CRITERIA = 'f1x'  # criteria can be ['mean_val_loss', 'mean_val_loss_binary', 'f1x', 'f1x_binary']

# print(POS_IDX)
