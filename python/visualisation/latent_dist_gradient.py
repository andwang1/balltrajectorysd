import matplotlib.pyplot as plt
import numpy as np
from exp_config import *
import os
from sklearn.neighbors import KDTree

# make font bigger
plt.rc('font', size=20)
def plot_latent_dist_gradient_in_dir(path, generate_images=True, save_path=None):

    NUM_NEIGHBOURS = 30

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

        DIST_FILE_NAME = f'distances{GEN_NUMBER}.dat'
        with open(DIST_FILE_NAME, "r") as f:
            lines = f.readlines()
        distances = np.array([float(i) for i in lines[-1][:-2].strip().split(",")])
        moved_indices = [int(i) for i in lines[5].strip().split()]

        fig = plt.figure(figsize=(15, 15))
        spec = fig.add_gridspec(1, 1)
        ax1 = fig.add_subplot(spec[:, :], aspect='equal', adjustable='box')

        max_value = np.max(np.abs(np.array([x, y])))
        plt.ylim([-max_value, max_value])
        plt.xlim([-max_value, max_value])
        x = np.array(x)
        y = np.array(y)

        points_data = np.concatenate((np.expand_dims(x, 1), np.expand_dims(y, 1)), axis=1)
        tree = KDTree(points_data)
        _, ind = tree.query(points_data, k=NUM_NEIGHBOURS)
        flattened_ind = np.ravel(ind)
        neighbour_dist_values = distances[flattened_ind].reshape(ind.shape)
        change_in_dist = np.abs(neighbour_dist_values[:, 0:1] - neighbour_dist_values[:, 1:])
        avg_change_in_dist = np.mean(change_in_dist, axis=1)


        # ax1 = fig.add_subplot()
        scatterplot = ax1.scatter(x, y, c=avg_change_in_dist)
        cbar = plt.colorbar(scatterplot)
        cbar.set_label(f'Change in trajectory distance for {NUM_NEIGHBOURS} nearest neighbours', rotation=270, labelpad=30)

        circ = plt.Circle((0, 0), radius=1, facecolor="None", edgecolor="black", linestyle="--", linewidth=2)
        ax1.add_patch(circ)

        plt.title(f"Latent Space - Gen {GEN_NUMBER} - Total Num. {len(x)} - % Moved {round(100 * (len(moved_indices) / len(x)), 1)}")
        plt.xlabel("Latent X")
        plt.ylabel("Latent Y")

        if save_path:
            os.chdir(save_path)

        plt.savefig(f"latent_dist_gradient_{GEN_NUMBER}.png")
        plt.close()


if __name__ == "__main__":
    plot_latent_dist_gradient_in_dir(
        "/home/andwang1/airl/results_balltrajectorysd_vae/gen3001_random1_fulllosstrue_beta1_extension0_lossfunc2_samplefalse_tsne1/2020-08-10_12_05_33_7736")
# "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/l2beta0nosample/results_balltrajectorysd_vae/gen6001_random1_fulllosstrue_beta0_extension0_lossfunc2_samplefalse/2020-07-21_01_26_48_639513")
