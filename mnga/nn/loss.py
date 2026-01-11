import numpy as np
from .module import Module
from ..autograd import huber_loss, log_softmax

class MSELoss(Module):
    def forward(self, pred, target):
        return ((pred - target) ** 2).mean()

class HuberLoss(Module):
    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
    
    def forward(self, pred, target, reduction=None):
        # Use the passed reduction if provided, else use the default set in __init__
        red = reduction if reduction is not None else self.reduction
        return huber_loss(pred, target, delta=self.delta, reduction=red)

class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        log_probs = log_softmax(pred)
        
        batch_size = pred.shape[0]
        
        if hasattr(target, 'data'):
            target = target.data
        if isinstance(target, list):
            target = np.array(target)

        idx = (np.arange(batch_size), target)
        selected_log_probs = log_probs[idx]
        
        return -selected_log_probs.mean()