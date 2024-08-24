import torch

a = torch.ones((2,5))
b = torch.ones((2,5))
t = torch.concat([a, b], dim=0)
print(t.shape)

