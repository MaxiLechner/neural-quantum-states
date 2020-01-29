from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hamiltonian import initialize_sutherland_1d

import jax.numpy as np

from time import time
from pathlib import Path
import warnings
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_bool("Complex_warning", True, "Surpress Complex warning")
flags.DEFINE_bool("pbc", True, "Periodic boundary conditions")
flags.DEFINE_float(
    "learning_rate",
    1e-02,
    "Learning rate for training. Either float or schedule.",
    short_name="lr",
)
flags.DEFINE_integer("print_every", 10, "Print results every n'th iteration")
flags.DEFINE_integer("batch_size", 100, "Batch size", short_name="bs")
flags.DEFINE_integer("num_spins", 10, "Number of spins", short_name="L")
flags.DEFINE_integer("epochs", 200, "Number of epochs")
flags.DEFINE_integer("seed", 0, "Seed for jax PRNG")
flags.DEFINE_integer("width", 12, "Width of the network")
flags.DEFINE_integer("filter_size", 3, "Size of the convolution filters")
flags.DEFINE_enum(
    "network",
    "small_net_1d",
    ["small_net_1d", "small_resnet_1d"],
    "NN from network.py",
    short_name="m",
)
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

    step, opt_state, key = initialize_sutherland_1d(
        FLAGS.width,
        FLAGS.filter_size,
        FLAGS.seed,
        FLAGS.num_spins,
        FLAGS.lr,
        FLAGS.batch_size,
        FLAGS.pbc,
        FLAGS.network,
    )

    gs_energy = -3.9

    E = []
    E_imag = []
    mag = []
    E_var = []
    Diff = []
    E0 = []
    sample = []
    lpsi = []
    lprob_i = []
    msk = []

    old_time = time()
    print("Step\tEnergy\tMagnetization\tVariance\ttime/step")
    print("---------------------------------------------------------")

    for i in range(FLAGS.epochs):
        opt_state, key, energy, e_imag, magnetization, var, e0, diff, s, logpsi, lprob, mask = step(
            i, opt_state, key
        )
        lprob_i.append(lprob)
        msk.append(mask)
        lpsi.append(logpsi)
        E0.append(e0)
        Diff.append(diff)
        sample.append(s)
        E.append(energy)
        E_imag.append(e_imag)
        mag.append(magnetization)
        E_var.append(var.real)
        if i % FLAGS.print_every == 0 and i > 0:
            new_time = time()
            print(
                "{}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.2f}".format(
                    i, energy, magnetization, var.real, new_time - old_time
                )
            )
            old_time = new_time

    directory = Path(FLAGS.filedir)
    subdir = "sutherland1d_pbc_{}_size_{}_bsize_{}_{}_lr_{}_epochs_{}_width_{}_fs_{}_seed_{}/".format(
        FLAGS.pbc,
        FLAGS.num_spins,
        FLAGS.batch_size,
        FLAGS.network,
        FLAGS.lr,
        FLAGS.epochs,
        FLAGS.width,
        FLAGS.filter_size,
        FLAGS.seed,
    )
    directory = directory / subdir

    if directory.is_dir():
        np.save(directory / "lprob_i", lprob_i)
        np.save(directory / "mask", msk)
        np.save(directory / "logpsi", lpsi)
        np.save(directory / "diff", Diff)
        np.save(directory / "e0", E0)
        np.save(directory / "sample", sample)
        np.save(directory / "energy", E)
        np.save(directory / "energy_imag", E_imag)
        np.save(directory / "magnetization", mag)
        np.save(directory / "energy_var", E_var)
        if FLAGS.pbc:
            np.save(directory / "exact_energy", gs_energy)
    else:
        directory.mkdir(parents=True)
        np.save(directory / "lprob_i", lprob_i)
        np.save(directory / "mask", msk)
        np.save(directory / "logpsi", lpsi)
        np.save(directory / "diff", Diff)
        np.save(directory / "e0", E0)
        np.save(directory / "sample", sample)
        np.save(directory / "energy", E)
        np.save(directory / "energy_imag", E_imag)
        np.save(directory / "magnetization", mag)
        np.save(directory / "energy_var", E_var)
        if FLAGS.pbc:
            np.save(directory / "exact_energy", gs_energy)


if __name__ == "__main__":
    app.run(main)
