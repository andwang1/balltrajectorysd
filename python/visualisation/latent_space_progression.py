import matplotlib.pyplot as plt
import numpy as np
from exp_config import *
import os

# make font bigger
plt.rc('font', size=20)

def plot_latent_space_in_dir(path, generate_images=True, save_path=None):
    os.chdir(path)
    files = os.listdir()

    # Find generation numbers from distances as we need the moved data
    generations = []
    for fname in files:
        if fname.startswith("distances") and ".dat" in fname:
            generations.append(fname.rstrip(r".dat")[len("distances"):])
    generations = sorted(int(gen) for gen in generations)

    num_generations = len(generations)
    # for plots across generations
    fig = plt.figure(figsize=(5 * num_generations, 5))
    spec = fig.add_gridspec(1, num_generations)
    axes = [fig.add_subplot(spec[0, i]) for i in range(num_generations)]
    plt.subplots_adjust(wspace=0.01)
    # hide ticks
    for ax in axes:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    for idx, GEN_NUMBER in enumerate(generations):
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
        axes[idx] = fig.add_subplot(spec[0, idx])#, aspect='equal', adjustable='box')
        max_value = np.max(np.abs(np.array([x, y])))
        # plt.ylim([-5.753798961639404, 5.753798961639404])
        # plt.xlim([-5.753798961639404, 5.753798961639404])
        axes[idx].set_ylim([-max_value, max_value])
        axes[idx].set_xlim([-max_value, max_value])
        # print(max_value)
        x = np.array(x)
        y = np.array(y)


        axes[idx].scatter(x[is_moved], y[is_moved], c="green", label="Moved")
        axes[idx].scatter(x[np.invert(is_moved)], y[np.invert(is_moved)], c="red", label="Not Moved")

        circ = plt.Circle((0, 0), radius=1, facecolor="None", edgecolor="black", linestyle="--", linewidth=2)
        axes[idx].add_patch(circ)

        # plt.title(f"Latent Space - Gen {GEN_NUMBER} - Total Num. {len(x)} - % Moved {round(100 * (len(x[is_moved]) / len(x)), 1)}")
        # plt.xlabel("Latent X")
        # plt.ylabel("Latent Y")
        # plt.legend()

        if save_path:
            os.chdir(save_path)

    plt.savefig(f"latent_space_progression.pdf")
    plt.savefig(f"latent_space_progression.png")
    plt.close()


if __name__ == "__main__":
    plot_latent_space_in_dir(
        "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/sne_nosampletrain_beta0/results_balltrajectorysd_vae/gen6001_random1_fulllosstrue_beta0_extension0_lossfunc2_samplefalse_tsne1/2020-08-04_03_51_15_231600")
# "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/l2beta0nosample/results_balltrajectorysd_vae/gen6001_random1_fulllosstrue_beta0_extension0_lossfunc2_samplefalse/2020-07-21_01_26_48_639513")
