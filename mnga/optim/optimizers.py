import numpy as np

class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad = None

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocities = [np.zeros_like(param.data) for param in self.parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
                
            grad = param.grad
            
            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                update = self.velocities[i]
            else:
                update = grad
            
            param.data -= self.lr * update

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(param.data) for param in self.parameters]
        self.v = [np.zeros_like(param.data) for param in self.parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
                
            grad = param.grad
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
