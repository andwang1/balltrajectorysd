import matplotlib.pyplot as plt
import time
import os
from exp_config import *

variant = "vae"
random = "0.4"
GEN_NUMBER = 6000
beta = "1"
extension = "0"
vae_loss = "fulllosstrue"

BASE_PATH = '/media/andwang1/SAMSUNG/MSC_INDIV/results_exp1/repeated_run1/L1/'
EXP_PATH = f'results_balltrajectorysd_{variant}/gen6001_random{random}_{vae_loss}/'
# EXP_PATH = f'results_balltrajectorysd_{variant}/gen6001_random{random}_{vae_loss}_beta{beta}_extension{extension}/'
FULL_PATH = BASE_PATH + EXP_PATH
os.chdir(FULL_PATH)

pids = [dir for dir in os.listdir() if os.path.isdir(os.path.join(FULL_PATH, dir))]
PID = pids[0] + "/"
os.chdir(BASE_PATH)
FILE_NAME = f'traj_{GEN_NUMBER}.dat'

FILE = BASE_PATH + EXP_PATH + PID + FILE_NAME

# PLOTTING PARAMETERS
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
    # The data
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

    f = plt.figure(figsize=(15, 10))
    spec = f.add_gridspec(4, 2)
    # both kwargs together make the box squared
    ax1 = f.add_subplot(spec[:2, :], aspect='equal', adjustable='box')
    ax1.set_ylim([ROOM_H, 0])
    ax1.set_xlim([0, ROOM_W])
    ax1.set_title("Trajectories")

    ax2 = f.add_subplot(spec[2, :])
    ax2.set_ylim([0, max(pred_error)])
    ax2.set_xlim([0, len_trajectory / 2])
    ax2.set_xlabel("Trajectory Step")
    ax2.yaxis.grid(True)
    ax2.set_title("L2 Error", loc="left")

    if vae_loss == "fulllosstrue":
        ax3 = f.add_subplot(spec[3, :])
        ax3.set_ylim([0, max(var)])
        ax3.set_xlim([0, len_trajectory / 2])
        ax3.set_xlabel("Trajectory Step")
        ax3.yaxis.grid(True)
        ax3.set_title("Decoder Variance", loc="left")

    # keep space between subplots
    plt.subplots_adjust(hspace=0.6)

    for i, j in zip(x_label, y_label):
        ax1.scatter(i, j, c="black")
        plt.pause(0.00000001)

    for index, (i, j, e, v) in enumerate(zip(x_pred, y_pred, pred_error, var)):
        ax1.scatter(i, j, c="red")
        ax2.scatter(index, e, s=4, c="black")
        if vae_loss == "fulllosstrue":
            ax3.scatter(index, v, s=4, c="green")
        plt.pause(0.00000001)

    time.sleep(PAUSE)
    plt.close()