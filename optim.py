from jax import jit


def step_init(
    energy_func, sample_func, grad_func, log_amplitude, data, opt_update, get_params
):
    @jit
    def step(i, opt_state, key):
        params = get_params(opt_state)
        key, sample = sample_func(params, data, key)
        energy = energy_func(params, sample)
        g = grad_func(params, sample, energy)
        var = energy_var(energy)
        return (
            opt_update(i, g, opt_state),
            key,
            energy.real.mean(),
            energy.imag.mean(),
            magnetization(sample),
            var,
        )

    return step
