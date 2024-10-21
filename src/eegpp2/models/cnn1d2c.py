import torch
from torch import nn

from .. import params


class MNAPooling1D(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.max_pooling = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.average_pooling = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        mx = self.max_pooling(x)
        avg = self.average_pooling(x)
        return torch.concat([mx, avg], dim=1)


class BiMaxPooling1D(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.max_pooling = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        mx1 = self.max_pooling(x)
        mx2 = self.max_pooling(-x)
        return torch.concat([mx1, mx2], dim=1)


class CNN1D2CModel(nn.Module):
    def __init__(self, n_class=params.NUM_CLASSES, n_base=16, flag=params.W_OUT, out_collapsed=True, n_conv=8,
                 window_size=5):
        super().__init__()
        self.n_class = n_class
        self.window_size = window_size
        self.type = "cnn1d2c"
        self.flag = flag
        self.chain1_layers = nn.ModuleList()
        self.chain2_layers = nn.ModuleList()
        self.chains = nn.ModuleList([self.chain1_layers, self.chain2_layers])
        # self.chains = [self.chain1_layers, self.chain2_layers]  # TorchScript can't convert `list`
        self.out_collapsed = out_collapsed

        base_dim = 1536

        for chain in self.chains:
            # chain = self.chains[i]
            layer1 = nn.Sequential(nn.Dropout(0.1),
                                   nn.Conv1d(1, n_base * 3, kernel_size=11, stride=4, padding=0),
                                   # nn.BatchNorm1d(n_base * 3),
                                   nn.LeakyReLU(),
                                   # nn.MaxPool1d(kernel_size=3, stride=2)
                                   BiMaxPooling1D(kernel_size=3, stride=2)
                                   )

            layer2 = nn.Sequential(nn.Conv1d(n_base * 3 * 2, n_base * 8, kernel_size=5, stride=1, padding=2),
                                   # nn.BatchNorm1d(n_base * 8),
                                   nn.LeakyReLU(),
                                   BiMaxPooling1D(kernel_size=3, stride=2))

            layer3 = nn.Sequential(nn.Conv1d(n_base * 8 * 2, n_base * 20, kernel_size=3, stride=1, padding=2),
                                   # nn.BatchNorm1d(n_base * 10),
                                   nn.LeakyReLU(),
                                   BiMaxPooling1D(kernel_size=3, stride=2)
                                   )

            layer4 = nn.Sequential(nn.Conv1d(n_base * 20 * 2, n_base * 8, kernel_size=3, stride=1, padding=2),
                                   # nn.BatchNorm1d(n_base * 8),
                                   nn.LeakyReLU(),
                                   BiMaxPooling1D(kernel_size=3, stride=2)
                                   )

            layer5 = nn.Sequential(nn.Conv1d(n_base * 8 * 2, n_base * 6, kernel_size=3, stride=1, padding=2),
                                   # nn.BatchNorm1d(n_base * 6),
                                   nn.LeakyReLU(),
                                   BiMaxPooling1D(kernel_size=3, stride=2)
                                   )
            chain.append(layer1)
            chain.append(layer2)
            chain.append(layer3)
            chain.append(layer4)
            chain.append(layer5)

            # self.fc1 = nn.Sequential(nn.Dropout(0.1), nn.Linear(2304, 320), nn.ReLU())
            fc1 = nn.Sequential(nn.Dropout(0.1), nn.Linear(base_dim * self.flag, 320), nn.LeakyReLU())
            chain.append(fc1)
            # self.fc1 = nn.Sequential(nn.Dropout(0.1), nn.Linear(1536, 320), nn.ReLU())

            # self.fc1 = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, 320), nn.ReLU())

        self.fc2 = nn.Sequential(nn.Linear(320 * 2, 320), nn.LeakyReLU(), nn.Linear(320, n_class * self.window_size))
        self.binary_cls = nn.Sequential(nn.Linear(n_class, 20),
                                        nn.ReLU(), nn.Linear(20, 2))
        self.binary_cls_join = nn.Sequential(nn.Linear(n_class * self.window_size + 320 * 2, 20),
                                             nn.ReLU(), nn.Linear(20, 2 * self.window_size))

    def forward(self, x):
        # print("X", x.shape)
        xis = []
        for i, chain in enumerate(self.chains):
            xi = x[:, i, :]
            xi = torch.unsqueeze(xi, 1)
            # xi = torch.squeeze(xi, 2)  # What does this line mean?
            for j, jlayer in enumerate(chain):
                if j < len(chain) - 1:
                    xi = jlayer(xi)
                else:
                    xi = xi.reshape(xi.size(0), -1)
                    xi = jlayer(xi)
                    xis.append(xi)
        out = torch.concat(xis, dim=-1)
        outx = out
        out = self.fc2(out)
        out = out.reshape(out.shape[0], self.window_size, -1)
        # out2 = torch.transpose(out, 1, 2)

        tmpx = torch.concat([outx, out.reshape(out.shape[0], -1)], dim=1)
        # print("TMPX: ", tmpx.shape)
        out2 = self.binary_cls_join(tmpx)
        # print("OUT2 cls join", out2.shape)
        out2 = out2.reshape((out2.shape[0], self.window_size, -1))
        # print("Final out2: ", out2.shape)
        return out, out2
