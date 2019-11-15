from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
from jax import random
import jax.numpy as np
from jax import jit
from jax.lax import fori_loop

from wavefunction import lpsi
from sampler import sample, sample_fori
from util import make_complex, apply_elementwise

import matplotlib.pyplot as plt
from functools import partial


def initialize_ising1d(batchSize, numSpins, network):
    M = (
        2 * 2
    )  # number of possible values each input can take times 2 as lax.conv only works with real valued weights
    FilterSize = 3
    model = network(M, FilterSize)
    net_init, net_apply = model
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    in_shape = (-1, numSpins, 1)
    _, net_params = net_init(subkey, in_shape)
    net_apply = jit(net_apply)
    data = np.zeros((batchSize, numSpins, 1), dtype=np.float32)
    return net_apply, net_params, data, key


@partial(jit, static_argnums=(0, 2))
def energy(net_apply, net_params, lpsi, state):
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


@partial(jit, static_argnums=(0, 2))
def energy_fori(net_apply, net_params, lpsi, state):
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

    def body_fun(i, loop_carry):
        a, b = loop_carry
        E = a - b[:, i] * b[:, i + 1] - amplitude_diff(state, i)
        return E, b

    logpsi = lpsi(net_apply, net_params, state)
    logpsi = logpsi[0] + logpsi[1] * 1j

    r = state.shape[1] - 1
    start_val = np.zeros(state.shape[0])
    start_val = start_val[..., None]
    start_val = start_val.astype("complex64")
    E, _ = fori_loop(0, r, body_fun, (start_val, state))
    return E


@jit
def energy_var(energy):
    emean = np.mean(energy)
    var = np.square(energy - emean)
    return np.mean(var)


@partial(jit, static_argnums=(0, 2))
def grad(net_apply, net_params, lpsi, state, energy):
    """computes the gradient (jacobian as lpsi returns two real numbers instead of one complex number)
    of the local energy lpsi by computing jac and multipliying it component wise with the local energy eloc"""
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
    mag = np.sum(state, axis=1)
    return np.mean(mag)


def callback(params, i, ax):
    E, mag, end_time, start_time = params
    print("iteration {} took {:.4f} secs.".format(i, end_time - start_time))
    plt.cla()
    ax.plot(E, label="Energy")
    ax.plot(mag, label="Magnetization")
    plt.legend()
    plt.draw()
    plt.pause(1.0 / 60.0)


# @partial(jit, static_argnums=(0,))
def step(i, net_apply, opt_update, get_params, opt_state, data):
    params = get_params(opt_state)
    s = sample(net_apply, params, i, data)
    e = energy(net_apply, params, lpsi, s)
    g = grad(net_apply, params, lpsi, s, e)
    var = energy_var(e)
    return opt_update(i, g, opt_state), e.real.mean(), magnetization(s), var


# @partial(jit, static_argnums=(0,))
def step_fori(i, net_apply, opt_update, get_params, opt_state, data, key):
    params = get_params(opt_state)
    key, s = sample_fori(net_apply, params, i, data, key)
    e = energy_fori(net_apply, params, lpsi, s)
    g = grad(net_apply, params, lpsi, s, e)
    var = energy_var(e)
    return opt_update(i, g, opt_state), key, e.real.mean(), magnetization(s), var


# @partial(jit, static_argnums=(0,))
def step2(i, net_apply, opt_update, get_params, opt_state, data):
    params = get_params(opt_state)
    s = sample(net_apply, params, i, data)
    e = energy_fori(net_apply, params, lpsi, s)
    g = grad(net_apply, params, lpsi, s, e)
    var = energy_var(e)
    return opt_update(i, g, opt_state), e.real.mean(), magnetization(s), var
