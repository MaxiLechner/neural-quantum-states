from ising1d import initialize_ising1d, step, callback
from network import net, resnet
import jax.numpy as np

from time import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    batchSize = 100
    numSpins = 10
    lr = 1e-02
    epochs = 300
    net_apply, net_params, data, key, opt_init, opt_update, get_params = initialize_ising1d(
        batchSize, numSpins, lr, resnet
    )

    gs_energy = 1 - 1 / (np.sin(np.pi / (2 * (2 * numSpins + 1))))
    E = []
    mag = []
    E_var = []
    Time = [time()]
    fig, ax = plt.subplots()
    plt.ion()
    plt.show(block=False)

    opt_state = opt_init(net_params)
    for i in range(epochs):
        opt_state, key, energy, magnetization, var = step(
            i, net_apply, opt_update, get_params, opt_state, data, key
        )
        E.append(energy)
        mag.append(magnetization)
        E_var.append(var.real)
        Time.append(time())
        callback((E, mag, Time, epochs, gs_energy), i, ax)

    ax.plot(E_var, label="Variance")
    plt.legend()
    plt.show(block=True)
