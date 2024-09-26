import torch
import torch.nn as nn
import torchvision

import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper parameters
input_size = 784  # 28 x 28
hidden_size = 100
num_classed = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

train_dataset=torchvision.datasets.MNIST(root='./data',train=True,
                                         transform=transforms.ToTensor(),download=True)

test_dataset=torchvision.datasets.MNIST(root='./data',train=False,
                                         transform=transforms.ToTensor())

train_loader=torchvision.datasets(dataset=train_dataset,batch_size=batch_size,shuffle=True)
