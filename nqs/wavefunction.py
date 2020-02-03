import jax.numpy as np
from jax import jit, vmap
from .util import real_to_complex

from functools import partial


def log_amplitude_init(net_apply):
    @jit
    def log_amplitude(net_params, data):
        """compute logpsi for a batch of samples. As the network returns
        the amplitude for both up and down states we need to pick the
        right amplitude by indexing according to the samples"""

        def index(x, y, i):
            xi = x[i]  # shape: (N,2)
            yi = y[i]  # shape: (N)
            arange = np.arange(xi.shape[0])
            return xi[arange, yi]

        arr = net_apply(net_params, data)
        arr = real_to_complex(arr)
        tc = np.exp(arr)
        nc = np.linalg.norm(tc, 2, axis=2, keepdims=True) ** 2

        # change representation from -1,1 to 0,1 for indexing purposes
        B, _, _ = data.shape
        idx = (data + 1) / 2
        idx = idx.astype(np.int32).squeeze()
        index = vmap(partial(index, arr, idx))
        vi = index(np.arange(B))[..., np.newaxis]
        logpsi = vi - 0.5 * np.log(nc)
        logpsi = np.sum(logpsi, axis=1)
        return np.real(logpsi), np.imag(logpsi)

    return log_amplitude
