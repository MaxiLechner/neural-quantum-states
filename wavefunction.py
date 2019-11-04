from jax import random, grad
import jax.numpy as np


def psi(predictions):
    """computes the wavefunction psi from the unormalized input vectors
    given by the output of the autoregressive model"""
    lnpsi, lnpsi_i = logpsi(predictions)
    psi_i = np.exp(lnpsi_i)
    psi = np.exp(lnpsi)
    p2 = np.prod(psi_i, axis=1)
    return psi, p2, psi_i


def logpsi(predictions):
    """computes the log-wavefunction from the unormalized input vectors
    given by the output of the autoregressive model"""
    l2norm = np.exp(predictions)
    l2norm = np.absolute(l2norm) ** 2
    l2norm = np.sum(l2norm, axis=2)
    l2norm = np.expand_dims(l2norm, axis=2)
    lnpsi_i = predictions - 0.5 * np.log(l2norm)
    lnpsi = np.sum(lnpsi_i, axis=1)
    return lnpsi, lnpsi_i


def sample(state, key, i, N):
    """draws samples from the wavefunction psi"""
    wf = psi(state)
    sqnorm = (np.linalg.norm(psi, axis=2) ** 2)[0][i]
    rand = random.bernoulli(key, sqnorm) * 2 - 1
    # print(sqnorm,rand)
    for i in range(N):
        wavefunction = psi(state)
        # print(vis)
        key, subkey = random.split(key)
        ss = sample(vis, key, i)
        print(ss)
        data[0][i] = rand
    return data
