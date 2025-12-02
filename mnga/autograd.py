import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _ctx=None):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = _ctx

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype
        
    @property
    def T(self):
        return self.transpose()

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self, grad=None):
        if self._ctx is None:
            return

        if grad is None:
            if self.data.size == 1:
                grad = np.ones_like(self.data)
            else:
                raise RuntimeError("grad must be specified for non-scalar tensor")

        self.grad = grad if self.grad is None else self.grad + grad

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v._ctx is not None:
                    for parent in v._ctx.parents:
                        build_topo(parent)
                topo.append(v)
        build_topo(self)

        self.grad = grad
        for node in reversed(topo):
            if not node.requires_grad:
                continue
            if node._ctx is not None:
                grads = node._ctx.backward(node._ctx, node.grad)
                if len(node._ctx.parents) == 1:
                    grads = [grads]
                for parent, parent_grad in zip(node._ctx.parents, grads):
                    if parent.requires_grad:
                        if parent.grad is None:
                            parent.grad = parent_grad
                        else:
                            parent.grad += parent_grad

    def zero_grad(self):
        self.grad = None

    def detach(self):
        return Tensor(self.data, requires_grad=False)

    def numpy(self):
        return self.data

    # Operations
    def __add__(self, other):
        return Add.apply(self, other)

    def __radd__(self, other):
        return Add.apply(other, self)

    def __mul__(self, other):
        return Mul.apply(self, other)

    def __rmul__(self, other):
        return Mul.apply(other, self)

    def __sub__(self, other):
        return Add.apply(self, Neg.apply(other))

    def __rsub__(self, other):
        return Add.apply(other, Neg.apply(self))

    def __neg__(self):
        return Neg.apply(self)

    def __truediv__(self, other):
        return Mul.apply(self, Pow.apply(other, -1))

    def __rtruediv__(self, other):
        return Mul.apply(other, Pow.apply(self, -1))

    def __pow__(self, power):
        return Pow.apply(self, power)

    def __matmul__(self, other):
        return MatMul.apply(self, other)

    def sum(self, axis=None, keepdims=False):
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        return Mean.apply(self, axis=axis, keepdims=keepdims)
        
    def transpose(self, *axes):
        return Transpose.apply(self, axes=axes)
        
    def reshape(self, *shape):
        return Reshape.apply(self, shape=shape)

    def __getitem__(self, idx):
        return GetItem.apply(self, idx=idx)

class Function:
    @classmethod
    def apply(cls, *args, **kwargs):
        ctx = cls()
        ctx.parents = [arg if isinstance(arg, Tensor) else Tensor(arg) for arg in args]
        ctx.saved_tensors = []
        ctx.kwargs = kwargs
        output_data = ctx.forward(ctx, *[p.data for p in ctx.parents], **kwargs)
        requires_grad = any(p.requires_grad for p in ctx.parents)
        return Tensor(output_data, requires_grad=requires_grad, _ctx=ctx)

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    def save_for_backward(self, *args):
        self.saved_tensors = args

# --- Operations ---

def unbroadcast(grad, shape):
    if grad.shape == shape:
        return grad
    
    ndims_added = grad.ndim - len(shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)
        
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
            
    return grad

class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x.shape, y.shape)
        return x + y

    @staticmethod
    def backward(ctx, grad_output):
        shape_x, shape_y = ctx.saved_tensors
        return unbroadcast(grad_output, shape_x), unbroadcast(grad_output, shape_y)

class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return unbroadcast(grad_output * y, x.shape), unbroadcast(grad_output * x, y.shape)
    
    def save_for_backward(self, *args):
        self.saved_tensors = args

class Neg(Function):
    @staticmethod
    def forward(ctx, x):
        return -x

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

class Pow(Function):
    @staticmethod
    def forward(ctx, x, power):
        ctx.save_for_backward(x, power)
        return x ** power

    @staticmethod
    def backward(ctx, grad_output):
        x, power = ctx.saved_tensors
        return grad_output * power * (x ** (power - 1)), None

class MatMul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x @ y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_output @ y.T
        grad_y = x.T @ grad_output
        return grad_x, grad_y
    
    def save_for_backward(self, *args):
        self.saved_tensors = args

class Sum(Function):
    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        ctx.save_for_backward(x.shape)
        ctx.axis = axis
        ctx.keepdims = keepdims
        return np.sum(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        input_shape, = ctx.saved_tensors
        axis = ctx.axis
        keepdims = ctx.keepdims
        
        if axis is None:
            grad = np.ones(input_shape) * grad_output
        else:
            if not keepdims:
                grad_output = np.expand_dims(grad_output, axis=axis)
            grad = np.ones(input_shape) * grad_output
        return grad

class Mean(Function):
    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        ctx.save_for_backward(x.shape)
        ctx.axis = axis
        ctx.keepdims = keepdims
        return np.mean(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        input_shape, = ctx.saved_tensors
        axis = ctx.axis
        keepdims = ctx.keepdims
        
        if axis is None:
            n = np.prod(input_shape)
            grad = np.ones(input_shape) * grad_output / n
        else:
            if not keepdims:
                grad_output = np.expand_dims(grad_output, axis=axis)
            n = input_shape[axis] if isinstance(axis, int) else np.prod([input_shape[i] for i in axis])
            grad = np.ones(input_shape) * grad_output / n
        return grad

class Transpose(Function):
    @staticmethod
    def forward(ctx, x, axes=None):
        if axes == (None,): axes = None
        ctx.axes = axes
        return np.transpose(x, axes)

    @staticmethod
    def backward(ctx, grad_output):
        axes = ctx.axes
        if axes is None:
            return np.transpose(grad_output)
        return np.transpose(grad_output, np.argsort(axes))

class Reshape(Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.save_for_backward(x.shape)
        return np.reshape(x, shape)

    @staticmethod
    def backward(ctx, grad_output):
        input_shape, = ctx.saved_tensors
        return np.reshape(grad_output, input_shape)

class GetItem(Function):
    @staticmethod
    def forward(ctx, x, idx):
        ctx.save_for_backward(x.shape, idx)
        return x[idx]

    @staticmethod
    def backward(ctx, grad_output):
        input_shape, idx = ctx.saved_tensors
        grad = np.zeros(input_shape)
        np.add.at(grad, idx, grad_output)
        return grad

class Maximum(Function):
    @staticmethod
    def forward(ctx, x, y):
        out = np.maximum(x, y)
        ctx.save_for_backward(x, y, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, y, out = ctx.saved_tensors
        mask_x = (x >= y)
        return grad_output * mask_x, grad_output * (~mask_x)
    
    def save_for_backward(self, *args):
        self.saved_tensors = args

class Exp(Function):
    @staticmethod
    def forward(ctx, x):
        out = np.exp(x)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        return grad_output * out
    
    def save_for_backward(self, *args):
        self.saved_tensors = args

class Log(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.log(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / x
    
    def save_for_backward(self, *args):
        self.saved_tensors = args

class Tanh(Function):
    @staticmethod
    def forward(ctx, x):
        out = np.tanh(x)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        return grad_output * (1 - out ** 2)
    
    def save_for_backward(self, *args):
        self.saved_tensors = args

def maximum(x, y):
    return Maximum.apply(x, y)

def exp(x):
    return Exp.apply(x)

def log(x):
    return Log.apply(x)

def tanh(x):
    return Tanh.apply(x)
