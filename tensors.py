import torch
import numpy as np

a=torch.ones(5)
print(a)
b=a.numpy()file:///home/chirag/Documents/GitHub/MAA_for_GoggleWS

print(type(b))
a.add_(1)
print(a)
print(type(b))