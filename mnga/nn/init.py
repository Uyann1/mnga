import numpy as np

def _calculate_fans(shape):
    if len(shape) < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    fan_in = shape[1]
    fan_out = shape[0]
    return fan_in, fan_out

def xavier_uniform(shape):
    fan_in, fan_out = _calculate_fans(shape)
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape).astype(np.float32)

def he_uniform(shape):
    fan_in, _ = _calculate_fans(shape)
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, shape).astype(np.float32)

def he_normal(shape):
    fan_in, _ = _calculate_fans(shape)
    std = np.sqrt(2 / fan_in)
    return np.random.normal(0, std, shape).astype(np.float32)

def orthogonal(shape, gain=1.0):
    if len(shape) < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = shape[0]
    cols = np.prod(shape[1:])
    
    flattened = np.random.normal(0.0, 1.0, (rows, cols))
    u, _, vt = np.linalg.svd(flattened, full_matrices=False)
    
    # Rescale the matrix
    q = u if u.shape == (rows, cols) else vt
    return (gain * q.reshape(shape)).astype(np.float32)

def uniform(shape, a, b):
    return np.random.uniform(a, b, shape).astype(np.float32)

def zeros(shape):
    return np.zeros(shape, dtype=np.float32)

def ones(shape):
    return np.ones(shape, dtype=np.float32)
