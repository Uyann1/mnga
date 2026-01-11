import numpy as np
from typing import List
from ..autograd import Tensor

class Optimizer:
    def __init__(self, parameters, lr: float = 0.001):
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = np.float32(lr)

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.zero_grad()
    
    def clip_grad_norm(self, max_norm: float) -> float:
        parameters = [p for p in self.parameters if p.grad is not None]

        if not parameters:
            return 0.0
        
        total_norm = np.sqrt(sum(np.sum(p.grad ** 2) for p in parameters))
        clip_coef = max_norm / (total_norm + 1e-6)

        if clip_coef < 1:
            for p in parameters:
                p.grad *= clip_coef
        return total_norm


class SGD(Optimizer):
    def __init__(self, parameters, lr: float = 0.001, momentum: float = 0.0):
        super().__init__(parameters, lr)
        self.momentum = np.float32(momentum)
        self.velocities = [np.zeros_like(param.data, dtype=np.float32) for param in self.parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
                
            grad = param.grad
            
            if self.momentum > 0:
                self.velocities[i] = (self.momentum * self.velocities[i] + grad).astype(np.float32)
                update = self.velocities[i]
            else:
                update = grad
            
            param.data -= self.lr * update

class Adam(Optimizer):
    def __init__(self, parameters, lr: float=0.001, beta1: float=0.9, beta2: float=0.999, epsilon: float=1e-8):
        super().__init__(parameters, lr)
        self.beta1 = np.float32(beta1)
        self.beta2 = np.float32(beta2)
        self.epsilon = np.float32(epsilon)
        self.m = [np.zeros_like(param.data, dtype=np.float32) for param in self.parameters]
        self.v = [np.zeros_like(param.data, dtype=np.float32) for param in self.parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
                
            grad = param.grad
            
            self.m[i] = (self.beta1 * self.m[i] + (1 - self.beta1) * grad).astype(np.float32)
            self.v[i] = (self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)).astype(np.float32)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class RMSProp(Optimizer):
    def __init__(self, parameters, lr: float=0.01, alpha: float=0.99, epsilon: float=1e-8):
        super().__init__(parameters, lr)
        self.alpha = np.float32(alpha)
        self.epsilon = np.float32(epsilon)
        self.square_avg = [np.zeros_like(param.data, dtype=np.float32) for param in self.parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            self.square_avg[i] = (self.alpha * self.square_avg[i] + (1 - self.alpha) * (grad ** 2)).astype(np.float32)
            avg = self.square_avg[i]
            
            param.data -= self.lr * grad / (np.sqrt(avg) + self.epsilon)
