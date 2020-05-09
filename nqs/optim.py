import jax.numpy as np
from jax import jit, grad, jvp
import jax

from functools import partial


def loss_init(log_amplitude):
    @jit
    def loss(net_params, config, energy):
        energy -= np.mean(energy)
        energy = energy.conj()
        lpsi = log_amplitude(net_params, config)
        return 2 * np.mean(np.real(energy * lpsi))

    return loss


@partial(jit, static_argnums=(0,))
def hvp(loss, params, batch, v):
    """Computes the hessian vector product Hv.

  This implementation uses forward-over-reverse mode for computing the hvp.

  Args:
    loss: function computing the loss with signature
      loss(params, batch).
    params: pytree for the parameters of the model.
    batch:  A batch of data. Any format is fine as long as it is a valid input
      to loss(params, batch).
    v: pytree of the same structure as params.

  Returns:
    hvp: array of shape [num_params] equal to Hv where H is the hessian.
  """

    def loss_fn(x):
        return loss(x, batch)

    f = grad(loss_fn)
    _, f_jvp = jax.linearize(f, params)
    out_tangent = f_jvp(v)
    return out_tangent

    # return jvp(grad(loss_fn), (params,), (v,))[1]


def step_init(
    energy_fn,
    sample_fn,
    loss_fn,
    energy_var,
    magnetization,
    log_amplitude,
    init_config,
    opt_update,
    get_params,
):
    @jit
    def loss(x, y):
        return -np.mean(np.real(log_amplitude(x, y)))

    @jit
    def step(i, opt_state, key):
        params = get_params(opt_state)
        key, config = sample_fn(params, init_config, key)
        energy = energy_fn(params, config)
        grad_loss = grad(loss_fn)(params, config, energy)
        var = energy_var(energy)
        mag = magnetization(config)
        # fisher = partial(hvp, loss, params, config)
        # grad_loss = jax.scipy.sparse.linalg.cg(fisher, grad_loss)[0]
        update = opt_update(i, grad_loss, opt_state)
        return update, key, energy, mag, var

    return step
