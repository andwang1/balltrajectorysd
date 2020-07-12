import matplotlib.pyplot as plt
import numpy as np
from exp_config import *
import os

GEN_NUMBER = 6000
step_size = 20

def plot_latent_space_in_dir(path, generate_images=True, save_path=None):
    os.chdir(path)

    FILE_NAME = f'archive_{GEN_NUMBER}.dat'
    with open(FILE_NAME, "r") as f:
        lines = f.readlines()

    # retrieve latent representations
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

    # trajectory data
    FILE = f'traj_{GEN_NUMBER}.dat'
    with open(FILE, 'r') as f:
        lines = f.readlines()[1:]

    # list of lists of recon, trajectories, losses
    plotting_data = []

    num_individuals = int(lines[-1].strip().split(",")[0])

    indiv_counter = 0
    line_number = 0
    file_length = len(lines)
    for i in range(num_individuals + 1):
        indiv_data = []
        while True:
            data = lines[line_number].strip().split(",")
            if int(data[0]) != i:
                data.append(indiv_data)
                break
            indiv_data.append([float(i) for i in data[2:]])
            line_number += 1
            if file_length == line_number:
                break
        plotting_data.append(indiv_data)

    # Plotting
    len_trajectory = len(indiv_data[0])

    # sorting along one dimension of the latent space, change to other coordinate for walking in the other axis
    sorted_indices = np.argsort(x)

    for index in sorted_indices[::step_size]:
        # change to other coordinate for walking in the other axis
        if y[index] > 0.1 or y[index] < -0.1:
            continue
        f = plt.figure(figsize=(10, 20))
        spec = f.add_gridspec(1, 2)
        ax1 = f.add_subplot(spec[0, 0], aspect='equal', adjustable='box')

        # latent space data
        max_value = np.max(np.abs(np.array([x, y])))
        plt.ylim([-max_value, max_value])
        plt.xlim([-max_value, max_value])

        x = np.array(x)
        y = np.array(y)

        ax1.scatter(x[is_moved], y[is_moved], c="green", label="Moved")
        ax1.scatter(x[np.invert(is_moved)], y[np.invert(is_moved)], c="red", label="Not Moved")
        ax1.scatter(x[index], y[index], c="yellow")

        circ = plt.Circle((0, 0), radius=1, facecolor="None", edgecolor="black", linestyle="--", linewidth=2)
        ax1.add_patch(circ)

        ax1.set_title(f"Latent Space - Gen {GEN_NUMBER} - Total Num. {len(x)} - % Moved {round(100 * (len(x[is_moved]) / len(x)), 1)}")
        ax1.set_xlabel("Latent X")
        ax1.set_ylabel("Latent Y")
        ax1.legend()

        # trajectory visualisation
        # The data
        indiv = plotting_data[index]
        prediction = indiv[0]
        pred_error = indiv[5]
        x_pred = prediction[::2]
        y_pred = prediction[1::2]

        # KL = indiv[2]
        var = indiv[3]
        # full_loss = indiv[1]

        label = indiv[4]
        x_label = label[::2]
        y_label = label[1::2]

        ax2 = f.add_subplot(spec[0, 1], aspect='equal', adjustable='box')
        ax2.set_ylim([ROOM_H, 0])
        ax2.set_xlim([0, ROOM_W])
        ax2.set_xlabel("Room X")
        ax2.set_ylabel("Room Y")
        ax2.set_title("Constructed Trajectory")

        ax2.scatter(x_pred, y_pred, c="black")
        plt.show()
        plt.close()



if __name__ == "__main__":
    plot_latent_space_in_dir(
        "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/l2/results_balltrajectorysd_vae/gen6001_random1_fulllosstrue_beta1_extension0_l2true/2020-06-23_08_38_43_138266")

    # good beta=0 visual "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/l1beta0/results_balltrajectorysd_vae/gen6001_random0.4_fulllosstrue_beta0_extension0_l2false/2020-06-24_16_44_25_172155")
