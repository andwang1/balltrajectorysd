import matplotlib.pyplot as plt
import time
from exp_config import *

GEN_NUMBER = 6000

BASE_PATH = '/home/andwang1/airl/balltrajectorysd/results_exp1/results_balltrajectorysd_ae/'
DIR_PATH = 'gen8000_pctrandom0.0/2020-06-01_18_15_45_81830/'
FILE_NAME = f'traj_{GEN_NUMBER}.dat'

FILE = BASE_PATH + DIR_PATH + FILE_NAME


# PLOTTING PARAMETERS
max_error = 10
PAUSE = 2


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

for indiv in plotting_data:
    f = plt.figure(figsize=(10, 5))
    spec = f.add_gridspec(3, 2)
    # both kwargs together make the box squared
    ax1 = f.add_subplot(spec[:2, :], aspect='equal', adjustable='box')
    ax1.set_ylim([ROOM_H, 0])
    ax1.set_xlim([0, ROOM_W])

    ax2 = f.add_subplot(spec[2, :])
    ax2.set_ylim([0, max_error])
    ax2.set_xlim([0, len_trajectory / 2])
    ax2.set_xlabel("Trajectory Step")
    ax2.yaxis.grid(True)

    ax1.set_title("Trajectories")
    ax2.set_title("L2 Error", loc="left")

    prediction = indiv[0]
    pred_error = indiv[1]
    x_pred = prediction[::2]
    y_pred = prediction[1::2]

    label = indiv[2]
    x_label = label[::2]
    y_label = label[1::2]

    print(prediction)
    print(label)

    for i, j in zip(x_label, y_label):
        ax1.scatter(i, j, c="black")
        plt.pause(0.00001)

    for index, (i, j, e) in enumerate(zip(x_pred, y_pred, pred_error)):
        ax1.scatter(i, j, c="red")
        ax2.scatter(index, e, s=4, c="black")
        plt.pause(0.001)

    time.sleep(PAUSE)
    plt.close()