# from jax import random
import jax.numpy as np


def psi(predictions):
    """computes the wavefunction psi from the unormalized input vectors
    given by the output of the autoregressive model"""
    l2norm = np.exp(predictions)
    l2norm = np.absolute(l2norm) ** 2
    l2norm = np.sum(l2norm, axis=2)
    l2norm = np.expand_dims(l2norm, axis=2)
    lnpsi_i = predictions - 0.5 * np.log(l2norm)
    psi_i = np.exp(lnpsi_i)
    lnpsi = np.sum(lnpsi_i, axis=1)
    psi = np.exp(lnpsi)
    return psi, psi_i


# def sample(psi, key, i):
#     """draws samples from the wavefunction psi"""
#     res = psi
#     sqnorm = (np.linalg.norm(res, axis=2) ** 2)[0][i]
#     rand = random.bernoulli(key, sqnorm) * 2 - 1
#     # print(sqnorm,rand)
#     for i in range(N):
#         vis = predictions(data)
#         # print(vis)
#         key, subkey = random.split(key)
#         ss = sample(vis, key, i)
#         print(ss)
#         data[0][i] = rand
#     return data
