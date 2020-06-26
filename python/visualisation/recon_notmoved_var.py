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
    percentage_moved = []

    for GEN_NUMBER in generations:
        if save_path:
            os.chdir(path)

        FILE_NAME = f'notmovedvar{GEN_NUMBER}.dat'

        with open(FILE_NAME, "r") as f:
            lines = f.readlines()
            mean_var.append(float(lines[2].strip()))

        MOVED_PCT_FILE_NAME = f'distances{GEN_NUMBER}.dat'
        with open(MOVED_PCT_FILE_NAME, "r") as f:
            lines = f.readlines()
        moved = int(lines[2].strip().split(",")[-1].split("/")[0])
        total = int(lines[2].strip().split(",")[-1].split("/")[1])
        percentage_moved.append(int(100 * moved / total))

    if generate_images:
        f = plt.figure(figsize=(5, 5))
        spec = f.add_gridspec(1, 1)
        ax1 = f.add_subplot(spec[0, 0])
        ln1 = ax1.plot(generations, mean_var, label="Variance", color="red")
        ax1.set_ylabel("Mean Recon. Variance")
        ax1.set_xlabel("Generations")

        ax2 = ax1.twinx()
        ln2 = ax2.plot(generations, percentage_moved, label="% Moved", color="blue")
        ax2.set_ylabel("%")
        ax2.set_ylim([0, 100])

        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='best')

        plt.title("Reconstruction Var. of No-Move solutions")
        plt.savefig("recon_not_moved_var.png")
        plt.close()

    data_dict = {"gen": generations, "NMV": mean_var}
    return data_dict


if __name__ == "__main__":
    plot_recon_not_moved_var_in_dir(
        "/home/andwang1/airl/imagesd/test_results/vistest/results_imagesd_vae/--number-gen=6001_--pct-random=0.2_--full-loss=true_--beta=1_--pct-extension=0_--loss-func=1_--sigmoid=false/2020-06-25_11_51_42_14672")
