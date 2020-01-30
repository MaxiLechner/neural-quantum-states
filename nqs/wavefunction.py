from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np
from jax import jit
from .util import real_to_complex


def log_amplitude_init(net_apply):
    @jit
    def log_amplitude(net_params, data):
        """compute logpsi for a batch of samples. As the network returns
        the amplitude for both up and down states we need to pick the
        right amplitude by indexing according to the samples"""
        arr = net_apply(net_params, data)
        arr = real_to_complex(arr)
        tc = np.exp(arr)
        nc = np.linalg.norm(tc, 2, axis=2, keepdims=True) ** 2

        idx = (data + 1) / 2
        idx = idx.astype("int32")
        B, N, _ = data.shape
        splits = np.split(arr, B)
        splits = [i.reshape(N, 2) for i in splits]
        isplits = np.split(idx, B)
        isplits = [i.reshape(N) for i in isplits]
        vi = np.stack([splits[j][np.arange(N), isplits[j]] for j in range(B)]).reshape(
            B, N, 1
        )

        logpsi = vi - 0.5 * np.log(nc)
        logpsi = np.sum(logpsi, axis=1)
        return np.real(logpsi), np.imag(logpsi)

    return log_amplitude
