import matplotlib.pyplot as plt
import numpy as np
from exp_config import *
import os
import seaborn as sns


def plot_recon_not_moved_var_in_dir(path, generate_images=True, save_path=None):
    os.chdir(path)
    files = os.listdir()

    # Find generation numbers
    generations = []
    for fname in files:
        if fname.startswith("notmovedvar") and ".dat" in fname:
            generations.append(fname.rstrip(r".dat")[len("notmovedvar"):])

    generations = sorted(int(gen) for gen in generations)

    # for plot across generations
    mean_var = []

    for GEN_NUMBER in generations:
        if save_path:
            os.chdir(path)

        FILE_NAME = f'notmovedvar{GEN_NUMBER}.dat'

        with open(FILE_NAME, "r") as f:
            lines = f.readlines()
            mean_var.append(float(lines[2].strip()))

    if generate_images:
        f = plt.figure(figsize=(3, 5))
        spec = f.add_gridspec(1, 1)
        ax1 = f.add_subplot(spec[0, 0])
        ln1 = ax1.plot(generations, mean_var, label="Variance", color="red")
        ax1.set_ylabel("Mean Recon. Variance")
        ax1.set_xlabel("Generations")
        ax1.legend()
        plt.title("Reconstruction Var. of No-Move solutions")
        plt.savefig("recon_not_moved_var.png")
        plt.close()

    data_dict = {"gen": generations, "NMV": mean_var}
    return data_dict


if __name__ == "__main__":
    plot_recon_not_moved_var_in_dir(
        "/home/andwang1/airl/balltrajectorysd/singularity/balltrajectorysd.sif/git/sferes2/build/exp/balltrajectorysd/balltrajectorysd_vae_2020-06-22_17_46_23_10922")
