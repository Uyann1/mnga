import numpy as np
from .module import Module
from ..autograd import sigmoid, relu, tanh

class ReLU(Module):
    def forward(self, x):
        return relu(x)

class Sigmoid(Module):
    def forward(self, x):
        return sigmoid(x)

class Tanh(Module):
    def forward(self, x):
        return tanh(x)