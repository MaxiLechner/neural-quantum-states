import jax.numpy as np
from jax import jit, vmap

from functools import partial


@jit
def log_amplitude(model, config):
    """compute logpsi for a batch of samples. As the network returns
    the amplitude for both up and down states we need to pick the
    right amplitude by indexing according to the samples"""

    def index_func(x, y, i):
        xi = x[i]  # shape: (N,2)
        yi = y[i]  # shape: (N)
        arange = np.arange(xi.shape[0])
        return xi[arange, yi]

    out = model(config)
    exp = np.exp(out)
    norm = np.linalg.norm(exp, 2, axis=2, keepdims=True) ** 2

    # change representation from -1,1 to 0,1 for indexing purposes
    B, _, _ = config.shape
    idx = (config + 1) / 2
    idx = idx.astype(np.int32).squeeze()
    index_func = vmap(partial(index_func, out, idx))
    vi = index_func(np.arange(B))[..., np.newaxis]
    logpsi = vi - 0.5 * np.log(norm)
    logpsi = np.sum(logpsi, axis=1)
    return logpsi
