from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np
from jax import jit, jacrev

from util import make_complex, apply_elementwise


def grad_init(log_amplitude):
    @jit
    def grad(net_params, state, energy):
        """computes the gradient (jacobian as log_amplitude returns two real numbers instead of one complex number) of the
        local energy log_amplitude by computing jac and multipliying it component wise with the local energy eloc"""
        eloc = energy.conj()
        eloc_mean = np.mean(eloc)
        eloc = eloc - eloc_mean
        jac = jacrev(log_amplitude)
        jac = jac(net_params, state)
        jac = make_complex(jac)
        jac = apply_elementwise(eloc, jac)
        return jac

    return grad


def step_init(
    energy_func,
    sample_func,
    grad_func,
    energy_var,
    magnetization,
    log_amplitude,
    init_batch,
    opt_update,
    get_params,
):
    @jit
    def step(i, opt_state, key):
        params = get_params(opt_state)
        key, sample = sample_func(params, init_batch, key)
        energy = energy_func(params, sample)
        grad = grad_func(params, sample, energy)
        var = energy_var(energy)
        return (
            opt_update(i, grad, opt_state),
            key,
            energy.real.mean(),
            energy.imag.mean(),
            magnetization(sample),
            var,
        )

    return step
