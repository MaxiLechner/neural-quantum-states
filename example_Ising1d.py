from ising1d import initialize_ising1d, step, callback
from network import net

from jax.experimental import optimizers

from time import time
import matplotlib.pyplot as plt
import pdb

if __name__ == "__main__":
    batchSize = 100
    numSpins = 10
    net_apply, net_params, key, data = initialize_ising1d(batchSize, numSpins, net)

    opt_init, opt_update, get_params = optimizers.adam(1e-02)

    E = []
    mag = []
    ratio = []

    fig, ax = plt.subplots()
    plt.ion()
    plt.show(block=False)

    opt_state = opt_init(net_params)
    for i in range(500):
        start_time = time()
        opt_state, key, energy, magnetization = step(
            i, key, net_apply, opt_update, get_params, opt_state, data
        )
        E.append(energy)
        mag.append(magnetization)
        end_time = time()
        if i % 25 == 0:
            callback((E, mag, end_time, start_time), i, ax)

    opt_init, opt_update, get_params = optimizers.adam(1e-03)
    # batchSize = 250
    # numSpins = 10
    # net_apply, net_params, key, data = initialize_ising1d(batchSize, numSpins, net)

    for i in range(2000):
        start_time = time()
        opt_state, key, energy, magnetization = step(
            i, key, net_apply, opt_update, get_params, opt_state, data
        )
        E.append(energy)
        mag.append(magnetization)
        end_time = time()
        if i % 25 == 0:
            callback((E, mag, end_time, start_time), i, ax)

    # pdb.set_trace()

    net_params = get_params(opt_state)
    plt.show(block=True)
