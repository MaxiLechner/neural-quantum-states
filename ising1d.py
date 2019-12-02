from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
from jax import random
import jax.numpy as np
from jax import jit
from jax.lax import fori_loop
from jax.experimental import optimizers

from network import small_resnet_1d
from wavefunction import lpsi
from sampler import check_sample, sample
from util import make_complex, apply_elementwise

import matplotlib.pyplot as plt
from functools import partial

import pdb


def initialize_ising1d(batchSize, numSpins, lr):
    M = (
        2 * 2
    )  # number of possible values each input can take times 2 as lax.conv only works with real valued weights
    FilterSize = 3
    model = small_resnet_1d(M, FilterSize)
    net_init, net_apply = model
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    in_shape = (-1, numSpins, 1)
    _, net_params = net_init(subkey, in_shape)
    net_apply = jit(net_apply)
    opt_init, opt_update, get_params = optimizers.adam(lr)
    data = np.zeros((batchSize, numSpins, 1), dtype=np.float32)
    return net_apply, net_params, data, key, opt_init, opt_update, get_params


@partial(jit, static_argnums=(0, 2))
def energy(net_apply, net_params, lpsi, state):
    @jit
    def amplitude_diff(state, i):
        """Compute amplitude ratio of logpsi and logpsi_fliped, where spin i has its
        sign fliped. As logpsi returns the real and the imaginary part seperately,
        we therefor need to recombine them into a complex valued array"""
        flip_i = np.ones(state.shape)
        flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, i], -1)
        fliped = state * flip_i
        logpsi_fliped = lpsi(net_apply, net_params, fliped)
        logpsi_fliped = logpsi_fliped[0] + logpsi_fliped[1] * 1j
        return np.exp(logpsi_fliped - logpsi)

    @jit
    def body_fun(i, loop_carry):
        a, b = loop_carry
        E = a - b[:, i] * b[:, i + 1] - amplitude_diff(b, i)
        return E, b

    logpsi = lpsi(net_apply, net_params, state)
    logpsi = logpsi[0] + logpsi[1] * 1j

    loop_start = 0
    loop_end = state.shape[1] - 1
    start_val = np.zeros(state.shape[0])
    start_val = start_val[..., None]
    start_val = start_val.astype("complex64")
    E, _ = fori_loop(loop_start, loop_end, body_fun, (start_val, state))
    E -= amplitude_diff(state, -1)
    return E


# # @partial(jit, static_argnums=(0, 2))
# def energy(net_apply, net_params, lpsi, state):
#     # @jit
#     def amplitude_diff(state, i):
#         """Compute apmplitude ratio of logpsi and logpsi_fliped, where i has its
#         sign fliped. As logpsi returns the real and the imaginary part seperately,
#         we therefor need to recombine them into a complex valued array"""
#         flip_i = np.ones(state.shape)
#         flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, i], -1)
#         fliped = state * flip_i
#         logpsi_fliped = lpsi(net_apply, net_params, fliped)
#         logpsi_fliped = logpsi_fliped[0] + logpsi_fliped[1] * 1j
#         return np.exp(logpsi_fliped - logpsi)

#     logpsi = lpsi(net_apply, net_params, state)
#     logpsi = logpsi[0] + logpsi[1] * 1j

#     loop_end = state.shape[1] - 1
#     start_val = np.zeros(state.shape[0])
#     start_val = start_val[..., None]
#     start_val = start_val.astype("complex64")

#     for i in range(loop_end):
#         start_val -= state[:, i] * state[:, i + 1] + amplitude_diff(state, i)
#         print(amplitude_diff(state, i))

#     start_val -= amplitude_diff(state, -1)
#     return start_val


@partial(jit, static_argnums=(0, 2))
def energy_heisenberg(net_apply, net_params, lpsi, state):
    @jit
    def amplitude_diff(state, i, j):
        """compute apmplitude ratio of logpsi and logpsi_fliped, where i and i+1
        have their sign fliped. As logpsi returns the real and the imaginary part
        seperately, we therefor need to recombine them into a complex valued array"""
        flip_i = np.ones(state.shape)
        flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, i], -1)
        flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, j], -1)
        fliped = state * flip_i
        logpsi_fliped = lpsi(net_apply, net_params, fliped)
        logpsi_fliped = logpsi_fliped[0] + logpsi_fliped[1] * 1j
        return np.exp(logpsi_fliped - logpsi)

    @jit
    def body_fun(i, loop_carry):
        E, mask, state = loop_carry
        E -= 0.5 * state[:, i] * state[:, i + 1] + 0.5 * mask[:, i] * amplitude_diff(
            state, i, i + 1
        )
        return E, mask, state

    mask = -state * np.roll(state, -1, axis=1) + 1
    logpsi = lpsi(net_apply, net_params, state)
    logpsi = logpsi[0] + logpsi[1] * 1j

    loop_start = 0
    loop_end = state.shape[1] - 1
    start_val = np.zeros(state.shape[0])
    start_val = start_val[..., None]
    start_val = start_val.astype("complex64")

    E, _, _ = fori_loop(loop_start, loop_end, body_fun, (start_val, mask, state))
    E -= 0.5 * state[:, -1] * state[:, 0]
    E -= 0.5 * mask[:, -1] * amplitude_diff(state, -1, 0)
    return E


@jit
def energy_var(energy):
    return np.var(energy)


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
    E, mag, Time, epochs, gs_energy = params
    epoch_mod = 100
    if i > 0 and i % epoch_mod == 0 or i == epochs - 1:
        print(
            "{} epochs took {:.4f} seconds.".format(
                epoch_mod, Time[i] - Time[i - epoch_mod]
            )
        )
        plt.cla()
        plt.axhline(gs_energy, label="Exact Energy", color="r")
        ax.plot(E, label="Energy")
        ax.plot(mag, label="Magnetization")
        plt.legend()
        plt.draw()
        plt.pause(1.0 / 60.0)


@partial(jit, static_argnums=(1, 2, 3))
def step(i, net_apply, opt_update, get_params, opt_state, data, key):
    params = get_params(opt_state)
    key, s = sample(net_apply, params, data, key)
    e = energy(net_apply, params, lpsi, s)
    g = grad(net_apply, params, lpsi, s, e)
    var = energy_var(e)
    return (opt_update(i, g, opt_state), key, e.real.mean(), magnetization(s), var, s)


@partial(jit, static_argnums=(1, 2, 3))
def step_heisenberg(i, net_apply, opt_update, get_params, opt_state, data, key):
    params = get_params(opt_state)
    key, sampl = sample(net_apply, params, data, key)
    # print(sampl.mean(), sampl.std())
    # if i == 0:
    # key, subkey = random.split(key)
    # probs = 0.5 * np.ones((100, 10, 1))
    # sampl = random.bernoulli(subkey, probs) * 2 - 1.0
    energy = energy_heisenberg(net_apply, params, lpsi, sampl)
    # print(logpsi.mean(), logpsi.std())
    # print(mask.sum() / 200.0, mask.mean(), mask.std())
    g = grad(net_apply, params, lpsi, sampl, energy)
    var = energy_var(energy)
    return (
        opt_update(i, g, opt_state),
        key,
        energy.real.mean(),
        energy.imag.mean(),
        magnetization(sampl),
        var,
    )
