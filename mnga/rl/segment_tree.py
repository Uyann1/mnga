import operator
from typing import Callable


class SegmentTree:
    def __init__(self, capacity: int, operation: callable, init_value: float):
        assert capacity > 0 and (capacity & (capacity - 1) == 0)
        self.capacity = capacity
        self.operation = operation
        self.init_value = init_value
        # Use Python List for faster single-element access
        self.tree = [init_value] * (2 * capacity)

    def operate(self, start: int = 0, end: int = None) -> float:
        if end is None: end = self.capacity
        start += self.capacity
        end += self.capacity
        res = self.init_value
        while start < end:
            if start & 1:
                res = self.operation(res, self.tree[start])
                start += 1
            if end & 1:
                end -= 1
                res = self.operation(res, self.tree[end])
            start >>= 1
            end >>= 1
        return res

    def __setitem__(self, idx: int, val: float):
        idx += self.capacity
        self.tree[idx] = val
        idx >>= 1
        # Use Python's built-in math inside the loop
        while idx >= 1:
            # Look up indices once for speed
            left = idx << 1
            self.tree[idx] = self.operation(self.tree[left], self.tree[left + 1])
            idx >>= 1

    def __getitem__(self, idx: int) -> float:
        return self.tree[self.capacity + idx]

    @property
    def total(self):
        return self.tree[1]

class SumSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        # Use operator.add (built-in) instead of np.add
        super().__init__(capacity=capacity, operation=operator.add, init_value=0.0)

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super().operate(start, end)
    
    def retrieve(self, upperbound: float) -> int:
        if upperbound > self.total:
            upperbound = self.total - 1e-12
        idx = 1
        while idx < self.capacity:
            left = idx << 1
            if self.tree[left] > upperbound:
                idx = left
            else:
                upperbound -= self.tree[left]
                idx = left + 1
        return idx - self.capacity
    
class MinSegmentTree(SegmentTree):

    def __init__(self, capacity: int):
        super().__init__(capacity=capacity, operation=min, init_value=float("inf"))

    def min(self, start: int = 0, end: int = 0) -> float:
        return super().operate(start, end)
