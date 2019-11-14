import jax
from jax import random
import jax.numpy as np
from jax import jit

# from jax.experimental import optimizers

# from network import net
from wavefunction import lpsi, make_complex, compute_probs, apply_elementwise

# from time import time
import matplotlib.pyplot as plt
from functools import partial


# from jax.lax import fori_loop


def initialize_ising1d(batchSize, numSpins, network):
    M = (
        2 * 2
    )  # number of possible values each input can take times 2 as lax.conv only works with real valued weights
    FilterSize = 3
    model = network(M, FilterSize)
    net_init, net_apply = model
    key = random.PRNGKey(1)
    key, subkey = random.split(key)
    in_shape = (-1, numSpins, 1)
    _, net_params = net_init(subkey, in_shape)
    net_apply = jit(net_apply)
    data = np.zeros((batchSize, numSpins, 1), dtype=np.float32)
    return net_apply, net_params, key, data


@partial(jit, static_argnums=(0,))
def sample(net_apply, net_params, key, data):
    for i in range(data.shape[1]):
        vi = net_apply(net_params, data)
        probs = compute_probs(vi)
        key, subkey = random.split(key)
        sample = random.bernoulli(subkey, probs[:, i, 1]) * 2 - 1.0
        sample = sample.reshape(data.shape[0], 1)
        data = jax.ops.index_update(data, jax.ops.index[:, i], sample)
    return key, data


@partial(jit, static_argnums=(0, 3))
def energy(net_apply, net_params, state, lpsi):
    @jit
    def amplitude_diff(state, i):
        """logpsi returns the real and the imaginary part seperately,
        we therefor need to recombine them into a complex valued array"""
        flip_i = np.ones(state.shape)
        flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, i], -1)
        fliped = state * flip_i
        logpsi_fliped = lpsi(net_apply, net_params, fliped)
        logpsi_fliped = logpsi_fliped[0] + logpsi_fliped[1] * 1j
        return np.exp(logpsi_fliped - logpsi)

    logpsi = lpsi(net_apply, net_params, state)
    logpsi = logpsi[0] + logpsi[1] * 1j
    E = 0
    for i in range(state.shape[1] - 1):
        E -= state[:, i] * state[:, i + 1] - amplitude_diff(state, i)
    E -= amplitude_diff(state, -1)
    return E


@partial(jit, static_argnums=(0, 3))
def grad(net_apply, net_params, state, lpsi, energy):
    eloc = energy.conj()
    eloc_mean = np.mean(eloc)
    eloc = eloc - eloc_mean
    jac = jax.jacrev(lpsi, argnums=1)
    jac = jac(net_apply, net_params, state)
    jac = make_complex(jac)
    jac = apply_elementwise(eloc, jac)
    return jac


@jit
def magnetization(state):
    E = np.sum(state, axis=1)
    return E


def callback(params, i, ax):
    E, mag, end_time, start_time = params
    print("iteration {} took {:.4f} secs.".format(i + 1, end_time - start_time))
    plt.cla()
    ax.plot(E, label="Energy")
    ax.plot(mag, label="Magnetization")
    plt.draw()
    plt.pause(1.0 / 60.0)


# @partial(jit, static_argnums=(0,))
def step(i, key, net_apply, opt_update, get_params, opt_state, data):
    params = get_params(opt_state)
    key, s = sample(net_apply, params, key, data)
    e = energy(net_apply, params, s, lpsi)
    g = grad(net_apply, params, s, lpsi, e)
    return opt_update(i, g, opt_state), e.real.mean(), magnetization(s).mean()
