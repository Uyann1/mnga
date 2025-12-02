import numpy as np
from .module import Module
from ..autograd import maximum

class MSELoss(Module):
    def forward(self, pred, target):
        return ((pred - target) ** 2).mean()

class HuberLoss(Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        diff = pred - target
        # We need absolute value, but we don't have abs in autograd yet.
        # Let's implement abs using maximum(x, -x)
        abs_diff = maximum(diff, -diff)
        
        quadratic = maximum(abs_diff, self.delta) # This logic is wrong for Huber.
        # Huber: 0.5 * x^2 if |x| < delta, else delta * (|x| - 0.5 * delta)
        # We need a conditional operation or masking.
        # Autograd doesn't support masking easily yet.
        # Let's stick to a simpler implementation or add masking to autograd.
        # Or we can just implement HuberLoss as a Function.
        
        # For now, let's implement a simplified version or just use MSE if Huber is too hard without masking.
        # But wait, I can implement HuberLossFunction.
        return HuberLossFunction.apply(pred, target, delta=self.delta)

from ..autograd import Function, Tensor

class HuberLossFunction(Function):
    @staticmethod
    def forward(ctx, pred, target, delta=1.0):
        diff = pred - target
        abs_diff = np.abs(diff)
        mask = abs_diff <= delta
        
        loss = np.where(mask, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta))
        
        ctx.save_for_backward(diff, mask, delta)
        return np.mean(loss)

    @staticmethod
    def backward(ctx, grad_output):
        diff, mask, delta = ctx.saved_tensors
        
        grad = np.where(mask, diff, delta * np.sign(diff))
        
        return grad * grad_output / diff.size, -grad * grad_output / diff.size, None
