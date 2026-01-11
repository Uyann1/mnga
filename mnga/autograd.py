import numpy as np

class Tensor:

    grad_enabled = True
    def __init__(self, data, requires_grad=False, _ctx=None, dtype=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if dtype is not None:
            self.data = data.astype(dtype, copy=False)
        elif np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.bool_):
            self.data = data
        else:
            self.data = data.astype(np.float32, copy=False)
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = _ctx

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype
        
    @property
    def T(self):
        return self.transpose()

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    def __str__(self):
        return f"Tensor({self.data})"

    def backward(self, grad=None, retain_graph=False):
        if self._ctx is None:
            return

        #  Initialize gradient
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar tensor")
            grad = np.ones_like(self.data)

        # Ensure raw numpy data    
        if hasattr(grad, 'data'):
            grad = grad.data

        # Initial accumulation
        self.grad = grad if self.grad is None else self.grad + grad

        #  Topological sort
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

        # Backward pass
        for node in reversed(topo):
            if node._ctx is None or node.grad is None:
                continue
            if not node.requires_grad:
                continue

            # --- THE CRITICAL FIX ---
            # Converts memoryview/slices back into a standard numpy array
            # This prevents "TypeError: bad operand type for unary -"
            node.grad = np.asarray(node.grad)          # -->> Look for redundancy
            # ------------------------

            grads = node._ctx.backward(node._ctx, node.grad)

            if not isinstance(grads, (tuple, list)):
                grads = (grads,)

            if hasattr(node._ctx, 'is_tensor_mask'):
                # We expect backward to return a grad for every arg. 
                # We only keep the ones where input was a Tensor.
                grads = [g for g, is_tensor in zip(grads, node._ctx.is_tensor_mask) if is_tensor]

            # Enforce contract
            assert len(grads) == len(node._ctx.parents), (
                f"{node._ctx.__class__.__name__}.backward returned "
                f"{len(grads)} grads for {len(node._ctx.parents)} parents"
            )

            for parent, parent_grad in zip(node._ctx.parents, grads):
                if parent.requires_grad and parent_grad is not None:
                    raw_grad = (
                        parent_grad.data
                        if hasattr(parent_grad, 'data')
                        else parent_grad
                    )

                    parent.grad = (
                        np.copy(raw_grad)
                        if parent.grad is None
                        else parent.grad + raw_grad
                    )

            # Free graph
            if not retain_graph:
                node._ctx = None

    def zero_grad(self):
        self.grad = None

    def detach(self):
        return Tensor(self.data, requires_grad=False)

    def numpy(self):
        return self.data
    
    def to_type(self, dtype):
        return Tensor(self.data.astype(dtype), requires_grad=False)

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
    
    def __abs__(x):
        return Abs.apply(x)

    def sum(self, axis=None, keepdims=False):
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        return Mean.apply(self, axis=axis, keepdims=keepdims)
        
    def argmax(self, axis=None, keepdims=False):
        return Tensor(np.argmax(self.data, axis=axis, keepdims=keepdims), requires_grad=False, dtype=np.int64)

    def max(self, axis=None, keepdims=False):
        return Max.apply(self, axis=axis, keepdims=keepdims)

    def item(self):
        if self.data.size != 1:
            raise ValueError("item() can only be called on tensors with one element")
        return self.data.item()

    def transpose(self, *axes):
        return Transpose.apply(self, axes=axes)
        
    def reshape(self, *shape):
        return Reshape.apply(self, shape=shape)
    
    def view(self, *shape):
        # 1. Handle flexible input: view(3, 4) or view((3, 4))
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        
        # 2. Check Contiguity (The PyTorch way)
        # If the data is not sitting in a simple, continuous line in memory
        if not self.data.flags['C_CONTIGUOUS']:
            try:
                # We create a temporary numpy view to see if the shape change is possible
                test_v = self.data.view()
                test_v.shape = shape # NumPy throws an error here if a copy is required
            except (AttributeError, ValueError):
                raise RuntimeError(
                    f"view size {shape} is not compatible with input tensor's size and stride. "
                    f"The tensor is non-contiguous. Use .reshape() if you want to allow a copy."
                )

        # 3. Apply the autograd Function
        return View.apply(self, shape=shape)
    
    def floor(self):
        return Floor.apply(self)
    
    def ceil(self):
        return Ceil.apply(self)
    
    def float(self):
        return Tensor(self.data.astype(np.float32), requires_grad=False)
    
    def long(self):
        return Tensor(self.data.astype(np.int64), requires_grad=False)
    
    def log(self):
        return Log.apply(self)
    
    def index_add_(self, dim, index, source):
        return IndexAdd.apply(self, dim, index, source)
    
    def clamp(self, min_value=None, max_value=None):
        return Clamp.apply(self, min_value=min_value, max_value=max_value)
    
    def index_add_(self, dim, index, source):
        """
        In-place version of index_add.
        Directly modifies self.data using NumPy's in-place scatter-add.
        """
        if dim < 0:
            dim += self.data.ndim
            
        idx_data = index.data if isinstance(index, Tensor) else index
        src_data = source.data if isinstance(source, Tensor) else source
        
        if not np.issubdtype(idx_data.dtype, np.integer):
            idx_data = idx_data.astype(np.int64)
            
        view = [slice(None)] * self.data.ndim
        view[dim] = idx_data
        
        np.add.at(self.data, tuple(view), src_data)
        
        return self
    
    @staticmethod
    def ones(shape, requires_grad=False, dtype=None):
        return Tensor(np.ones(shape), requires_grad=requires_grad, dtype=dtype)
    
    @staticmethod
    def ones_like(tensor, requires_grad=False, dtype=None):
        return Tensor(np.ones_like(tensor.data), requires_grad=requires_grad, dtype=dtype or tensor.dtype)
    
    @staticmethod
    def zeros(shape, requires_grad=False, dtype=None):
        return Tensor(np.zeros(shape), requires_grad=requires_grad, dtype=dtype)
    
    @staticmethod
    def zeros_like(tensor, requires_grad=False, dtype=None):
        return Tensor(np.zeros_like(tensor.data), requires_grad=requires_grad, dtype=dtype or tensor.dtype)
    
    @staticmethod
    def empty(shape, requires_grad=False, dtype=np.float32):
        return Tensor(np.empty(shape), requires_grad=requires_grad, dtype=dtype)
    
    @staticmethod
    def empty_like(tensor, requires_grad=False, dtype=None):
        return Tensor(np.empty_like(tensor.data), requires_grad=requires_grad, dtype=dtype or tensor.dtype)
    
    @staticmethod
    def random(shape, requires_grad=False, dtype=None):
        """ Uniform distributin [0, 1)"""
        return Tensor(np.random.random(shape), requires_grad=requires_grad, dtype=dtype)
    
    @staticmethod
    def random_like(tensor, requires_grad=False, dtype=None):
        return Tensor(np.random.random(tensor.shape), requires_grad=requires_grad, dtype=dtype or tensor.dtype)

    @staticmethod
    def randomint(low, high, shape, requires_grad=False, dtype=np.int64):
        return Tensor(np.random.randint(low, high, shape), requires_grad=requires_grad, dtype=dtype)
    
    @staticmethod
    def randomint_like(tensor, low, high, requires_grad=False, dtype=np.int64):
        return Tensor(np.random.randint(low, high, tensor.shape), requires_grad=requires_grad, dtype=dtype)
    
    @staticmethod
    def randomstdn(shape, requires_grad=False, dtype=None):
        """ Standard normal distribution (mean=0, std=1) """
        return Tensor(np.random.standard_normal(shape), requires_grad=requires_grad, dtype=dtype)
    
    @staticmethod
    def randomstdn_like(tensor, requires_grad=False, dtype=None):
        return Tensor(np.random.standard_normal(tensor.shape), requires_grad=requires_grad, dtype=dtype or tensor.dtype)
    
    @staticmethod
    def randomperm(n, requires_grad=False, dtype=np.int64):
        """ Random permutation of integers from 0 to n-1"""
        return Tensor(np.random.permutation(n), requires_grad=requires_grad, dtype=dtype)
    
    @staticmethod
    def linspace(start, stop, steps, requires_grad=False, dtype=None):
        return Tensor(np.linspace(start, stop, steps), requires_grad=requires_grad, dtype=dtype)
    
    @staticmethod
    def arange(start, stop=None, step=1, requires_grad=False, dtype=None):
        return Tensor(np.arange(start, stop, step), requires_grad=requires_grad, dtype=dtype)

    def __getitem__(self, idx):
        return GetItem.apply(self, idx=idx)

class no_grad:
    """Context-manager that disables gradient calculation."""        
    def __enter__(self):
        # 1. Save the previous state of gradient tracking
        self.prev = Tensor.grad_enabled
        # 2. Disable gradient tracking
        Tensor.grad_enabled = False
        
    def __exit__(self, exc_type, exc_value, traceback):
        # 3. Restore the previous state
        Tensor.grad_enabled = self.prev

def _arg_to_raw(v):
    if isinstance(v, Tensor): 
        return v.data
    if isinstance(v, (int, float, bool)):
        return np.array(v, dtype=np.float32)
    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=np.float32)
    return v

def _kwarg_to_raw(v):
    if isinstance(v, Tensor): 
        return v.data
    if isinstance(v, (int, float, bool, str, tuple, type(None))):
        return v
    return np.asarray(v, dtype=np.float32)
# ----------------------------------------------------------------------------------------

class Function:
    @classmethod
    def apply(cls, *args, **kwargs):
        # 1. Enforce the Rule: No Tensors in kwargs!
        # This saves you from the "Gradient Swap" nightmare.
        for k, v in kwargs.items():
            if isinstance(v, Tensor):
                raise ValueError(
                    f"Found Tensor in kwargs ('{k}'). "
                    "In this engine, all Tensors must be passed as positional arguments "
                    "to ensure correct gradient ordering."
                )

        # 2. Check gradients (Now only need to check args)
        tensors_in_args = [a for a in args if isinstance(a, Tensor)]
        
        # Simplified: We don't need to check kwargs for requires_grad anymore
        needs_grad = Tensor.grad_enabled and any(t.requires_grad for t in tensors_in_args)

        arg_data = [_arg_to_raw(a) for a in args]
        # Kwargs are just raw data now
        kwarg_data = {k: _kwarg_to_raw(v) for k, v in kwargs.items()}

        if not needs_grad:
            output_data = cls.forward(None, *arg_data, **kwarg_data)
            return Tensor(output_data, requires_grad=False)

        ctx = cls()
        
        # 3. Mask and Parents are now guaranteed to align perfectly
        ctx.is_tensor_mask = [isinstance(a, Tensor) for a in args]
        ctx.parents = tensors_in_args 
        ctx.saved_tensors = []
        
        output_data = cls.forward(ctx, *arg_data, **kwarg_data)
        
        return Tensor(output_data, requires_grad=True, _ctx=ctx)

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    def save_for_backward(self, *args):
        self.saved_tensors = args

def unbroadcast(grad, shape):
    # 1. If grad is already a Tensor, get its data
    grad = np.asarray(grad.data if hasattr(grad, 'data') else grad, dtype=np.float32)

    if grad.shape == shape:
        return grad

    # Case 2: Target is a scalar (0-d array)
    # Sum over all axes to reduce to scalar
    if shape == ():
        return grad.sum()

    # Case 3: Broadcasted dimensions handling
    # (Existing logic to sum out broadcasted dims)
    ndim_grad = grad.ndim
    ndim_shape = len(shape)
    
    diff = ndim_grad - ndim_shape
    if diff > 0:
        grad = grad.sum(axis=tuple(range(diff)))
        
    # Sum over axes where shape is 1 but grad is > 1 (broadcasted)
    keep_axes = tuple(i for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)) if s_dim == 1 and g_dim > 1)
    if keep_axes:
        grad = grad.sum(axis=keep_axes, keepdims=True)
        
    return grad

class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        if ctx is not None:
            ctx.save_for_backward(x.shape, y.shape)
        return x + y

    @staticmethod
    def backward(ctx, grad_output):
        shape_x, shape_y = ctx.saved_tensors
        return unbroadcast(grad_output, shape_x), unbroadcast(grad_output, shape_y)

class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        if ctx is not None:
            ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return unbroadcast(grad_output * y, x.shape), unbroadcast(grad_output * x, y.shape)

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
        if ctx is not None:
            ctx.save_for_backward(x, power)
        return x ** power

    @staticmethod
    def backward(ctx, grad_output):
        x, power = ctx.saved_tensors
        return grad_output * power * (x ** (power - 1))

class MatMul(Function):
    @staticmethod
    def forward(ctx, x, y):
        if ctx is not None:
            ctx.save_for_backward(x, y)
        return x @ y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        y_T = y.swapaxes(-1, -2)
        x_T = x.swapaxes(-1, -2)
        grad_x = grad_output @ y_T
        grad_y = x_T @ grad_output
        if grad_x.shape != x.shape:
            grad_x = unbroadcast(grad_x, x.shape)
        if grad_y.shape != y.shape:
            grad_y = unbroadcast(grad_y, y.shape)
        return grad_x, grad_y

class Sum(Function):
    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        if ctx is not None:
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
        if ctx is not None:
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
        if not axes:
            axes = None
        if ctx is not None:
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
        if ctx is not None:
            ctx.save_for_backward(x.shape)
        return np.reshape(x, shape)

    @staticmethod
    def backward(ctx, grad_output):
        input_shape, = ctx.saved_tensors
        return np.reshape(grad_output, input_shape)

class GetItem(Function):
    @staticmethod
    def forward(ctx, x, idx):
        # Helper to clean indices: Extract data from Tensors and ensure integers
        def clean_idx(i):
            if isinstance(i, Tensor):
                i = i.data
            # If it's a numpy array of floats, cast it to int64
            if isinstance(i, np.ndarray) and np.issubdtype(i.dtype, np.floating):
                return i.astype(np.int64)
            return i

        if isinstance(idx, tuple):
            raw_idx = tuple(clean_idx(i) for i in idx)
        else:
            raw_idx = clean_idx(idx)

        if ctx is not None:
            ctx.save_for_backward(x.shape, raw_idx)
            
        return x[raw_idx]

    @staticmethod
    def backward(ctx, grad_output):
        input_shape, raw_idx = ctx.saved_tensors
        grad = np.zeros(input_shape)
        # Using np.add.at is crucial for handling duplicate indices in a batch
        np.add.at(grad, raw_idx, grad_output)
        return grad

class Max(Function):
    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        if ctx is not None:
            ctx.axis = axis
            ctx.keepdims = keepdims
        values = np.max(x, axis=axis, keepdims=keepdims)
        if ctx is not None:
            ctx.save_for_backward(x, values)
        return values
    
    @staticmethod
    def backward(ctx, grad_output):
        x, values = ctx.saved_tensors
        axis = ctx.axis
        keepdims = ctx.keepdims

        if axis is not None and not keepdims:
            values_expanded = np.expand_dims(values, axis=axis)
            grad_output_expanded = np.expand_dims(grad_output, axis=axis)
        
        else:
            values_expanded = values
            grad_output_expanded = grad_output
        
        mask = (x == values_expanded).astype(np.float32)
        return (mask * grad_output_expanded) / np.sum(mask, axis=axis, keepdims=True)

class Maximum(Function):
    @staticmethod
    def forward(ctx, x, y):
        out = np.maximum(x, y)
        if ctx is not None:
            ctx.save_for_backward(x, y, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, y, out = ctx.saved_tensors
        mask_x = (x >= y)
        return grad_output * mask_x, grad_output * (~mask_x)

class Abs(Function):
    @staticmethod
    def forward(ctx, x):
        # We only save shapes if we have a context (needs_grad = True)
        if ctx is not None:
            # Save the sign of x for the backward pass
            ctx.save_for_backward(np.sign(x))
        return np.abs(x)

    @staticmethod
    def backward(ctx, grad_output):
        sign, = ctx.saved_tensors
        # The gradient of |x| is sign(x)
        return grad_output * sign

class Exp(Function):
    @staticmethod
    def forward(ctx, x):
        out = np.exp(x)
        if ctx is not None:
            ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        return grad_output * out

class Log(Function):
    @staticmethod
    def forward(ctx, x):
        if ctx is not None:
            ctx.save_for_backward(x)
        return np.log(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / x

class Clamp(Function):
    @staticmethod
    def forward(ctx, x, min_value=None, max_value=None):
        if ctx is not None:
            ctx.min_value = min_value
            ctx.max_value = max_value
            ctx.save_for_backward(x)
        return np.clip(x, min_value, max_value)
    
    @staticmethod 
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        min_val = ctx.min_value if ctx.min_value is not None else -np.inf
        max_val = ctx.max_value if ctx.max_value is not None else np.inf
        
        mask = (x >= min_val) & (x <= max_val)
        return grad_output * mask

class View(Function):
    @staticmethod
    def forward(ctx, x, shape):
        # 1. Normalize shape to a tuple of integers
        if isinstance(shape, (int, np.integer)):
            shape = (int(shape),)
        elif isinstance(shape, np.ndarray):
            shape = tuple(shape.astype(int).flatten())
        else:
            shape = tuple(int(s) for s in shape)

        if ctx is not None:
            ctx.save_for_backward(x.shape)
            
        return x.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        x_shape, = ctx.saved_tensors
        # FIX: Remove the ", None". 
        # You only have 1 parent (x), so you only return 1 gradient.
        return grad_output.reshape(x_shape)

class Floor(Function):
    @staticmethod
    def forward(ctx, x):
        return np.floor(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return np.zeros_like(grad_output)

class Ceil(Function):
    @staticmethod
    def forward(ctx, x):
        return np.ceil(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return np.zeros_like(grad_output)

class IndexAdd(Function):
    @staticmethod
    def forward(ctx, x, dim, index, source):
        # x, index, and source are now raw NumPy arrays
        # dim is a raw Python int
        out = x.copy()
        
        # Ensure index is integer type for NumPy indexing
        if np.issubdtype(index.dtype, np.floating):
            index = index.astype(np.int64)
            
        view = [slice(None)] * x.ndim
        view[dim] = index # This works now because dim is an int!
        
        np.add.at(out, tuple(view), source)
        
        if ctx is not None:
            # We save the shape and index for backward
            ctx.save_for_backward(x.shape, dim, index, source.shape)
            
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_shape, dim, index, source_shape = ctx.saved_tensors
        
        # grad_x is same as grad_output
        grad_x = grad_output
        
        # grad_source needs to be extracted from the indexed positions
        view = [slice(None)] * len(x_shape)
        view[dim] = index
        grad_source = grad_output[tuple(view)].reshape(source_shape)
        
        # We return None for dim and index because they aren't differentiable
        return grad_x, None, None, grad_source


class Sigmoid_(Function):
    @staticmethod
    def forward(ctx, x):
        out = 1 / (1 + np.exp(-x))
        if ctx is not None:
            ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        return grad_output * out * (1 - out)

class ReLU_(Function):
    @staticmethod
    def forward(ctx, x):
        mask = x > 0
        if ctx is not None:
            ctx.save_for_backward(mask)
        return np.maximum(0, x)
    
    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask

class Tanh_(Function):
    @staticmethod
    def forward(ctx, x):
        out = np.tanh(x)
        if ctx is not None:
            ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        return grad_output * (1 - out ** 2)

class Softmax(Function):
    @staticmethod
    def forward(ctx, x, axis=-1):
        if ctx is not None:
            ctx.axis = axis
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        out = e_x / np.sum(e_x, axis=axis, keepdims=True)
        if ctx is not None:
            ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        axis = ctx.axis

        term1 = grad_output * out
        term2 = np.sum(term1, axis=axis, keepdims=True)
        return term1 - out * term2

class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, x, axis=-1):
        if ctx is not None:
            ctx.axis = axis
        # Stable log_softmax: x - log(sum(exp(x)))
        # Use logsumexp trick: log(sum(exp(x))) = m + log(sum(exp(x-m)))
        # where m = max(x)
        m = np.max(x, axis=axis, keepdims=True)
        shifted_x = x - m
        exp_x = np.exp(shifted_x)
        sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
        log_sum_exp = np.log(sum_exp_x)
        
        output = shifted_x - log_sum_exp
        
        # Save output for backward. output is log_softmax(x)
        if ctx is not None:
            ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        axis = ctx.axis
        
        # Backward of log_softmax(x) w.r.t x:
        # grad_input = grad_output - sum(grad_output) * softmax(x)
        
        softmax_output = np.exp(output)
        
        sum_grad = np.sum(grad_output, axis=axis, keepdims=True)
        return grad_output - sum_grad * softmax_output

class HuberLossFunction(Function):
    @staticmethod
    def forward(ctx, pred, target, delta=1.0, reduction='mean'):
        # pred and target are raw numpy arrays here
        diff = pred - target
        abs_diff = np.abs(diff)
        mask = abs_diff <= delta

        loss = np.where(mask, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta))

        if ctx is not None:
            ctx.save_for_backward(diff, mask, delta)
            ctx.reduction = reduction
            
        if reduction == 'mean':
            return np.mean(loss)
        elif reduction == 'none':
            return loss
        return np.sum(loss)
    
    @staticmethod
    def backward(ctx, grad_output):
        diff, mask, delta = ctx.saved_tensors
        reduction = ctx.reduction

        # Basic gradient calculation
        grad = np.where(mask, diff, delta * np.sign(diff))

        # Handle reduction scaling
        if reduction == 'mean':
            grad = grad * grad_output / diff.size
        else:
            # For 'none', grad_output will be a tensor of the same shape as grad
            grad = grad * grad_output

        return grad, -grad

def maximum(x, y):
    return Maximum.apply(x, y)

def exp(x):
    return Exp.apply(x)

def log(x):
    return Log.apply(x)

def clamp(x, min_value=None, max_value=None):
    return Clamp.apply(x, min_value=min_value, max_value=max_value)

def tanh(x):
    return Tanh_.apply(x)

# Helper function
def huber_loss(pred, target, delta=1.0, reduction='mean'):
    return HuberLossFunction.apply(pred, target, delta=delta, reduction=reduction)

def sigmoid(x):
    return Sigmoid_.apply(x)

def relu(x):
    return ReLU_.apply(x)

def softmax(x, axis=-1):
    return Softmax.apply(x, axis=axis)

def log_softmax(x, axis=-1):
    return LogSoftmax.apply(x, axis=axis)

def mean(x, axis=None, keepdims=False):
    return Mean.apply(x, axis=axis, keepdims=keepdims)