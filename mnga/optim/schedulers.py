import numpy as np
import math

class _LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.base_lrs = [optimizer.lr] # Assuming single LR for now based on Optimizer implementation
        self.last_epoch = last_epoch
        
        # Initialize epoch if not already done
        if last_epoch == -1:
            self.step()
        else:
             self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        values = self.get_lr()
        # Update optimizer LR
        # MNGA Optimizer currently has a single self.lr attribute
        # We will assume for now we are updating that. 
        # Future/More robust: support param_groups like PyTorch
        self.optimizer.lr = values[0] 
        self._last_lr = values

class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [self.optimizer.lr]
        return [self.optimizer.lr * self.gamma]

class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        return [self.optimizer.lr * self.gamma]

class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        if (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [self.optimizer.lr + (self.base_lrs[0] - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2]
        
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (self.optimizer.lr - self.eta_min) + self.eta_min]
