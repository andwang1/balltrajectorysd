import matplotlib.pyplot as plt
import numpy as np
from exp_config import *
import os


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

        is_moved = np.array([False] * len(x))
        is_moved[moved_indices] = True

        fig = plt.figure(figsize=(15, 15))
        max_value = np.max(np.abs(np.array([x, y])))
        plt.ylim([-max_value, max_value])
        plt.xlim([-max_value, max_value])

        x = np.array(x)
        y = np.array(y)

        ax1 = fig.add_subplot()
        ax1.scatter(x[is_moved], y[is_moved], c="green", label="Moved")
        ax1.scatter(x[np.invert(is_moved)], y[np.invert(is_moved)], c="red", label="Not Moved")

        circ = plt.Circle((0, 0), radius=1, facecolor="None", edgecolor="black", linestyle="--", linewidth=2)
        ax1.add_patch(circ)

        plt.title(f"Latent Space - Gen {GEN_NUMBER} - Total Num. {len(x)} - % Moved {round(100 * (len(x[is_moved]) / len(x)), 1)}")
        plt.xlabel("Latent X")
        plt.ylabel("Latent Y")
        plt.legend()

        if save_path:
            os.chdir(save_path)

        plt.savefig(f"latent_space_{GEN_NUMBER}.png")
        plt.close()


if __name__ == "__main__":
    plot_latent_space_in_dir(
        "/home/andwang1/airl/balltrajectorysd/results_box2d_exp1/box2dtest/vistest/2020-06-19_19_12_49_126106")
