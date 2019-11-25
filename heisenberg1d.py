from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax

import jax.numpy as np
from jax import jit
from jax.lax import fori_loop

from functools import partial


# @partial(jit, static_argnums=(0, 2))
def energy(net_apply, net_params, lpsi, state):
    # @jit
    def amplitude_diff(state, i):
        """compute apmplitude ratio of logpsi and logpsi_fliped, where i and i+1
        have their sign fliped. As logpsi returns the real and the imaginary part
        seperately, we therefor need to recombine them into a complex valued array"""
        flip_i = np.ones(state.shape)
        flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, i], -1)
        flip_i = jax.ops.index_update(flip_i, jax.ops.index[:, i + 1], -1)
        fliped = state * flip_i
        logpsi_fliped = lpsi(net_apply, net_params, fliped)
        logpsi_fliped = logpsi_fliped[0] + logpsi_fliped[1] * 1j
        return np.exp(logpsi_fliped - logpsi)

    def conditional_apply(x, cond, i):
        return jax.lax.cond(cond, x, amplitude_diff, x, lambda _: 0)

    cond = np.roll(state, -1, axis=1) != 1

    # @jit
    def body_fun(i, loop_carry):
        a, b = loop_carry
        E = a - b[:, i] * b[:, i + 1] - 2 * conditional_apply(state, cond, i)
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
