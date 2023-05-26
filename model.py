from hypParam import *
from torch.nn import Module, Conv2d, ReLU, Sequential, Flatten, Linear, Tanh

class ConvBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.block = Sequential(
            Conv2d(in_channels, out_channels, kernel_size, stride),
            ReLU(inplace=True)
        )

    def forward(self, X):
        return self.block(X)

class Agent(Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.layers = Sequential(
            ConvBlock(3, fm[0], 5, 2),
            ConvBlock(fm[0], fm[1], 5, 2),
            ConvBlock(fm[1], fm[2], 5, 2),
            ConvBlock(fm[2], fm[3], 3, 1),
            ConvBlock(fm[3], fm[4], 3, 1),
            Flatten(),
            Linear(1152, fc[0]), ReLU(inplace=True),
            Linear(fc[0], fc[1]), ReLU(inplace=True),
            Linear(fc[1], fc[2]), ReLU(inplace=True),
            Linear(fc[2], 1),
            Tanh()
        )

    def forward(self, X):
        return self.layers(X)
