# Using Lightning Fabric

# Interface shape

- inp_shape = [batch, 3, (W_out * MAX_SEQ_LEN)]
- out1_shape = [batch, W_out, 7] --> just take 6 standard labels
- out2_shape = [batch, W_out, 2] --> represents for star prediction

# Turnoff channel

ids = [0, 1, 2]

1. turnoff in combine model

- inp_turnoff: inp[:, id, :] = 0

2. turnoff in discrete model

- skip signal

# Model input shape

1. CNN1D
2. FFT
3. Transformer
4. CNN1D_NC
5. FFT_NC
6. Transformer_NC
7. STFT
8. STFT_NC
