import jax.numpy as np
from jax import jit, grad
from jax.lax import fori_loop

from .wavefunction import log_amplitude


@jit
def loss(model, config, energy):
    energy -= np.mean(energy)
    energy = energy.conj()
    lpsi = log_amplitude(model, config)
    return 2 * np.mean(np.real(energy * lpsi))


@jit
def isgo_loss(model, config, energy, weights, num_steps):
    energy -= np.mean(weights * energy)
    energy = energy.conj()
    lpsi = weights * log_amplitude(model, config)
    return 2 * num_steps * np.mean(np.real(energy * lpsi))


def step_init(energy_fn, sample_fn, energy_var, magnetization):
    @jit
    def isgo_step(optimizer, key):
        @jit
        def isgo_body(i, carry):
            _, optimizer = carry
            model = optimizer.target
            energy = energy_fn(model, config)
            updated_weights = probs(model, config)
            weights = np.exp(updated_weights - init_weights)
            grad_loss = grad(isgo_loss)(model, config, energy, weights, num_steps)
            opt_update = optimizer.apply_gradient(grad_loss)
            return energy, opt_update

        @jit
        def probs(model, config):
            logpsi = log_amplitude(model, config)
            psi = np.exp(logpsi)
            probs = np.square(np.abs(psi))
            return probs

        num_steps = 5
        model = optimizer.target
        key, config = sample_fn(model, key)
        energy = energy_fn(model, config)
        init_weights = probs(model, config)
        energy, opt_update = fori_loop(0, num_steps, isgo_body, (energy, optimizer))

        var = energy_var(energy)
        mag = magnetization(config)

        return opt_update, key, energy, mag, var

    return isgo_step


# def step_init(energy_fn, sample_fn, energy_var, magnetization):
#     @jit
#     def step(optimizer, key):
#         model = optimizer.target
#         key, config = sample_fn(model, key)
#         energy = energy_fn(model, config)
#         grad_loss = grad(loss)(model, config, energy)
#         var = energy_var(energy)
#         mag = magnetization(config)
#         opt_update = optimizer.apply_gradient(grad_loss)
#         return opt_update, key, energy, mag, var

#     return step
