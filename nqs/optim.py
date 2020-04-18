import jax.numpy as np
from jax import jit, grad


def loss_init(energy_func, log_amplitude):
    @jit
    def loss(net_params, config, energy):
        energy -= np.mean(energy)
        energy = energy.conj()
        lpsi = log_amplitude(net_params, config)
        return 2 * np.mean(np.real(energy * lpsi))

    return loss


def step_init(
    energy_func,
    sample_func,
    loss_func,
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
        grad_loss = grad(loss_func)(params, config, energy)
        var = energy_var(energy)
        mag = magnetization(config)
        update = opt_update(i, grad_loss, opt_state)
        return (update, key, energy.real.mean(), energy.imag.mean(), mag, var)

    return step
