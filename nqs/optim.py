import jax.numpy as jnp
from jax import jit, grad

from .wavefunction import log_amplitude


@jit
def loss(model, config, energy):
    energy -= jnp.mean(energy)
    energy = energy.conj()
    lpsi = log_amplitude(model, config)
    return 2 * jnp.mean(jnp.real(energy * lpsi))


def step_init(energy_fn, learning_rate_fn, energy_var, magnetization):
    @jit
    def step(optimizer, key):
        model = optimizer.target
        key, config = model.sample(key=key)
        energy = energy_fn(model, config)
        grad_loss = grad(loss)(model, config, energy)
        var = energy_var(energy)
        mag = magnetization(config)
        lr = learning_rate_fn(optimizer.state.step)
        opt_update = optimizer.apply_gradient(grad_loss, learning_rate=lr)
        return opt_update, key, energy, mag, var, lr

    return step
