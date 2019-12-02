from ising1d import step_heisenberg
from network import small_net_1d

import jax.numpy as np
from jax import jit, random
from jax.experimental import optimizers

from time import time
from pathlib import Path
import warnings
from absl import app, flags

import matplotlib.pyplot as plt


FLAGS = flags.FLAGS
flags.DEFINE_bool("Complex_warning", True, "Surpress Complex warning")
flags.DEFINE_float("learning_rate", 1e-02, "Learning rate for training")
flags.DEFINE_integer("batch_size", 100, "Batch size")
flags.DEFINE_integer("num_spins", 10, "Number of spins")
flags.DEFINE_integer("epochs", 200, "Number of epochs")
flags.DEFINE_integer("seed", 0, "Seed for jax PRNG")
flags.DEFINE_integer("width", 12, "Width of the model")
flags.DEFINE_integer("filter_size", 3, "Size of the convolution filters")
flags.DEFINE_string(
    "filedir", "notebooks/results/", "Directory where data is saved.", short_name="f"
)
# flags.mark_flag_as_required('filename')


def main(unused_argv):
    if FLAGS.Complex_warning:
        warnings.filterwarnings(
            "ignore",
            message="Casting complex values to real discards the imaginary part",
        )
    model = small_net_1d(FLAGS.width, FLAGS.filter_size)
    net_init, net_apply = model
    key = random.PRNGKey(FLAGS.seed)
    key, subkey = random.split(key)
    in_shape = (-1, FLAGS.num_spins, 1)
    _, net_params = net_init(subkey, in_shape)
    net_apply = jit(net_apply)
    opt_init, opt_update, get_params = optimizers.adam(
        optimizers.polynomial_decay(FLAGS.learning_rate, 10, 0.0001, 3)
    )
    data = np.zeros((FLAGS.batch_size, FLAGS.num_spins, 1), dtype=np.float32)

    gs_energy = FLAGS.num_spins * (1 / 4 - np.log(2))
    E = []
    E_imag = []
    mag = []
    E_var = []
    Time = [time()]
    # _, ax = plt.subplots()
    # plt.ion()
    # plt.show(block=False)

    opt_state = opt_init(net_params)
    print_every = 10
    old_time = time()
    print("Step\tEnergy\tMagnetization\tVariance\ttime/step")
    print("---------------------------------------------------------")

    for i in range(FLAGS.epochs):
        opt_state, key, energy, e_imag, magnetization, var = step_heisenberg(
            i, net_apply, opt_update, get_params, opt_state, data, key
        )
        E.append(energy)
        E_imag.append(e_imag)
        mag.append(magnetization)
        E_var.append(var.real)
        Time.append(time())
        if i % print_every == 0 and i > 0:
            new_time = time()
            print(
                "{}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.2f}".format(
                    i, energy, magnetization, var.real, new_time - old_time
                )
            )
            old_time = new_time
    #         pars = get_params(opt_state)
    #         plt.cla()
    #         _, ax = plt.subplots(1, 5, figsize=(30, 3))
    #         for i in range(len(ax)):
    #             ax[i].hist(pars[i * 2][0].flatten())
    #             # ax[i].set_title("layer", i * 2)
    #         # plt.legend()
    #         plt.draw()
    #         plt.pause(1.0 / 60.0)

    # plt.legend()
    # plt.show(block=True)
    print("exact energy: ", gs_energy)

    directory = Path(FLAGS.filedir)
    subdir = "heisenberg1d_size_{}_bsize_{}_resnet1d_lr_{}_epochs_{}_width_{}_fs_{}_seed_{}/".format(
        FLAGS.num_spins,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.epochs,
        FLAGS.width,
        FLAGS.filter_size,
        FLAGS.seed,
    )
    directory = directory / subdir

    if directory.is_dir():
        np.save(directory / "exact_energy", gs_energy)
        np.save(directory / "energy", E)
        np.save(directory / "energy_imag", E_imag)
        np.save(directory / "magnetization", mag)
        np.save(directory / "energy_var", E_var)
    else:
        directory.mkdir(parents=True)
        np.save(directory / "exact_energy", gs_energy)
        np.save(directory / "energy", E)
        np.save(directory / "energy_imag", E_imag)
        np.save(directory / "magnetization", mag)
        np.save(directory / "energy_var", E_var)


if __name__ == "__main__":
    app.run(main)
