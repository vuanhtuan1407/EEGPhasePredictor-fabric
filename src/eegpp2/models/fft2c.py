import torch
from torch import nn

from .. import params


class FFT2CModel(nn.Module):
    def __init__(self, n_class=params.NUM_CLASSES, window_size=params.W_OUT, flag=1, out_collapsed=True):
        super().__init__()
        self.n_class = n_class
        self.window_size = window_size
        self.flag = flag
        self.out_collapsed = out_collapsed
        self.type = "fft2c"
        self.chain1_layers = nn.ModuleList()
        self.chain2_layers = nn.ModuleList()
        self.chains = [self.chain1_layers, self.chain2_layers]
        self.fc1 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(params.MAX_SEQ_SIZE * self.window_size, 320),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(320 * 2, 320),
            nn.LeakyReLU(),
            nn.Linear(320, n_class * self.window_size)
        )
        self.binary_cls_join = nn.Sequential(
            nn.Linear(n_class * self.window_size + 320 * 2, 20),
            nn.ReLU(),
            nn.Linear(20, 2 * self.window_size)
        )

    def forward(self, x):
        xis = []
        for i in range(2):
            xi = x[:, i, :]
            # xi = torch.squeeze(xi, 2)
            # print(xi.shape)
            # xi = xi.reshape(xi.shape[0], -1)
            xi = torch.fft.fft(xi)
            # print(xi.shape)
            xi = torch.sqrt(xi.real ** 2 + xi.imag ** 2)
            xi = self.fc1(xi)
            xis.append(xi)

        out = torch.concat(xis, dim=-1)
        outx = out
        out = self.fc2(out)
        out = out.reshape(-1, self.window_size, self.n_class)
        # out2 = torch.transpose(out, 1, 2)

        tmpx = torch.concat([outx, out.reshape(out.shape[0], -1)], dim=1)
        # print("TMPX: ", tmpx.shape)
        out2 = self.binary_cls_join(tmpx)
        # print("OUT2 cls join", out2.shape)
        out2 = out2.reshape((-1, self.window_size, 2))
        # print("Final out2: ", out2.shape)
        return out, out2
