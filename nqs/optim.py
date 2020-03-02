import jax.numpy as np
from jax import jit

# from jax.lax import fori_loop

from .util import make_complex, apply_elementwise


def grad_init(grad_psi):
    @jit
    def grad(net_params, config, energy):
        """computes the gradient (jacobian as log_amplitude returns two real numbers instead of one complex number) of the
        local energy log_amplitude by computing jac and multipliying it component wise with the local energy eloc"""
        eloc = energy.conj()
        eloc_mean = np.mean(eloc)
        eloc = eloc - eloc_mean
        jac = grad_psi(net_params, config)
        jac = make_complex(jac)
        jac = apply_elementwise(eloc, jac)
        return jac

    # @jit
    # def isgo_grad(net_params, config, energy, weights):
    #     eloc = energy.conj()
    #     eloc_mean = np.mean(weights * eloc)
    #     eloc = eloc - eloc_mean
    #     eloc = weights * eloc
    #     jac = grad_psi(net_params, config)
    #     jac = make_complex(jac)
    #     jac = apply_elementwise(eloc, jac)
    #     return jac

    # return isgo_grad
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
        key, config = sample_func(params, init_batch, key)
        energy = energy_func(params, config)
        grad = grad_func(params, config, energy)
        var = energy_var(energy)
        mag = magnetization(config)
        update = opt_update(i, grad, opt_state)
        return (update, key, energy.real.mean(), energy.imag.mean(), mag, var)

    # @jit
    # def isgo_step(i, opt_state, key):
    #     @jit
    #     def isgo_body(i, carry):
    #         _, opt_state = carry
    #         params = get_params(opt_state)
    #         energy = energy_func(params, config)
    #         updated_weights = probs(params, config)
    #         weights = np.exp(updated_weights - init_weights)
    #         energy = weights * energy
    #         grad = grad_func(params, config, energy, weights)
    #         update = opt_update(i, grad, opt_state)
    #         return energy, update

    #     @jit
    #     def probs(params, config):
    #         logpsi = log_amplitude(params, config)
    #         logpsi = logpsi[0] + logpsi[1] * 1j
    #         psi = np.exp(logpsi)
    #         probs = np.square(np.abs(psi))
    #         return probs

    #     num_steps = 1
    #     params = get_params(opt_state)
    #     key, config = sample_func(params, init_batch, key)
    #     init_weights = probs(params, config)

    #     energy = energy_func(params, config)
    #     grad = grad_func(params, config, energy, np.ones(init_weights.shape))
    #     update = opt_update(i, grad, opt_state)

    #     energy, update = fori_loop(0, num_steps - 1, isgo_body, (energy, update))

    #     var = energy_var(energy)
    #     mag = magnetization(config)
    #     return (update, key, energy.real.mean(), energy.imag.mean(), mag, var)

    # return isgo_step
    return step
