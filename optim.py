import matplotlib.pyplot as plt
from time import time

# from ising1d import step

from jax.experimental import optimizers


def optim(sample, energy, grad, key):
    def train(sample, energy, grad, key):

        # @partial(jit, static_argnums=(0,))
        def step(i, key, opt_state):
            params = get_params(opt_state)
            key, state = sample(key)
            energy = energy(params, state)
            g = grad(params, state, energy)
            E.append(energy.real.mean())
            mag.append(magnetization(state).mean())
            return opt_update(i, g, opt_state)

        opt_init, opt_update, get_params = optimizers.adam(lr)
        E = []
        mag = []

        fig, ax = plt.subplots()
        plt.ion()
        plt.show(block=False)

        opt_state = opt_init(net_params)
        for i in range(numIt):
            start_time = time()
            opt_state = step(i, key, opt_state)
            end_time = time()
            if i % 25 == 0 and plotting is True:
                callback(ax, (E, mag, end_time, start_time), i)

        net_params = get_params(opt_state)
        plt.show(block=True)
        return net_params, E, mag

    def callback(ax, params, i):
        E, mag, end_time, start_time = params
        print("iteration {} took {:.4f} secs.".format(i + 1, end_time - start_time))
        plt.cla()
        ax.plot(E, label="Energy")
        ax.plot(mag, label="Magnetization")
        plt.draw()
        plt.pause(1.0 / 60.0)

    return train
