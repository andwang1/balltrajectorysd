import matplotlib.pyplot as plt
import numpy as np
from exp_config import *
import os
# make font bigger
plt.rc('font', size=20)

def plot_latent_space_in_dir(path, save_path=None):
    os.chdir(path)
    files = os.listdir()

    # Find generation numbers from distances as we need the moved data
    generations = []
    for fname in files:
        if fname.startswith("distances") and ".dat" in fname:
            generations.append(fname.rstrip(r".dat")[len("distances"):])
    generations = sorted(int(gen) for gen in generations)

    for GEN_NUMBER in generations:
        if save_path:
            os.chdir(path)

        FILE_NAME = f'archive_{GEN_NUMBER}.dat'
        with open(FILE_NAME, "r") as f:
            lines = f.readlines()

        x = []
        y = []
        for line in lines:
            data = line.strip().split()
            x.append(float(data[1]))
            y.append(float(data[2]))

        MOVED_INDICES_FILE_NAME = f'distances{GEN_NUMBER}.dat'
        with open(MOVED_INDICES_FILE_NAME, "r") as f:
            lines = f.readlines()
        moved_indices = [int(i) for i in lines[5].strip().split()]

        x = np.array(x)
        y = np.array(y)

        is_moved = np.array([False] * len(x))
        is_moved[moved_indices] = True

        fig = plt.figure(figsize=(15, 15))
        plt.ylim([-4, 4])
        plt.xlim([-4, 4])

        ax1 = fig.add_subplot()
        ax1.scatter(x[is_moved], y[is_moved], c="green", label="Moved")
        ax1.scatter(x[np.invert(is_moved)], y[np.invert(is_moved)], c="red", label="Not Moved")

        circ = plt.Circle((0, 0), radius=1, facecolor="None", edgecolor="blue", linestyle="--")
        ax1.add_patch(circ)

        plt.title(f"Latent Space - Gen {GEN_NUMBER} - Total Num. {len(lines)}")
        plt.xlabel("Latent X")
        plt.ylabel("Latent Y")
        plt.legend()

        if save_path:
            os.chdir(save_path)

        plt.savefig(f"latent_space_{GEN_NUMBER}.pdf")
        plt.close()


if __name__ == "__main__":
    plot_latent_space_in_dir(
        "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/l2nosample/results_balltrajectorysd_vae/gen6001_random1_fulllosstrue_beta1_extension0_lossfunc2_samplefalse/2020-07-10_10_54_42_3109092")

