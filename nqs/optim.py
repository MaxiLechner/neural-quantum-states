import jax.numpy as np
from jax import jit, grad

from .wavefunction import log_amplitude


@jit
def loss(model, config, energy):
    energy -= np.mean(energy)
    energy = energy.conj()
    lpsi = log_amplitude(model, config)
    return 2 * np.mean(np.real(energy * lpsi))


def step_init(energy_fn, sample_fn, energy_var, magnetization):
    @jit
    def step(optimizer, key):
        model = optimizer.target
        key, config = sample_fn(model, key)
        energy = energy_fn(model, config)
        grad_loss = grad(loss)(model, config, energy)
        var = energy_var(energy)
        mag = magnetization(config)
        opt_update = optimizer.apply_gradient(grad_loss)
        return opt_update, key, energy, mag, var

    return step
