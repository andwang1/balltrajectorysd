import matplotlib.pyplot as plt
import time
import os
import numpy as np
from exp_config import *

GEN_NUMBER = 6000
FILE_NAME = f'undist_traj_{GEN_NUMBER}.dat'
BUCKET_FILE_NAME = f'distances{GEN_NUMBER}.dat'
BUCKET_IDX = 102

FULL_PATH = "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/l2nosampletrain/results_balltrajectorysd_vae/gen6001_random0_fulllossfalse_beta1_extension0_lossfunc2_samplefalse"

PIDs = [name for name in os.listdir(FULL_PATH) if os.path.isdir(os.path.join(FULL_PATH, name))]

f = plt.figure(figsize=(5, 5))
spec = f.add_gridspec(1, 1)
# both kwargs together make the box squared
ax1 = f.add_subplot(spec[0, 0], aspect='equal', adjustable='box')
ax1.set_ylim([ROOM_H, 0])
ax1.set_xlim([0, ROOM_W])
ax1.set_title("All Trajectories")

for dir in PIDs:
    os.chdir(FULL_PATH)
    FILE = FULL_PATH + "/" + dir +  "/" + FILE_NAME
    BUCKET_FILE = FULL_PATH + "/" + dir +  "/" + BUCKET_FILE_NAME
    with open(FILE, 'r') as f:
        lines = f.readlines()

    # list of lists of recon, trajectories, losses
    plotting_data = []

    for line in lines:
        plotting_data.append([float(i.strip()) for i in line.split(",")])

    with open(BUCKET_FILE, "r") as f:
        lines = f.readlines()
    for line in lines:
        data = line.split(",")
        if data[0].strip() != str(BUCKET_IDX):
            continue
        indices = [int(i) for i in data[1:]]
        break
    print(indices)
    plotting_data = [data for i, data in enumerate(plotting_data) if i in indices]



    for idx, indiv in enumerate(plotting_data):
        if idx % 100 == 0:
            print(f"Processing {idx}th trajectory")
        # The data
        x_label = indiv[::2]
        y_label = indiv[1::2]

        # Plot
        ax1.plot(x_label, y_label, color="gray", alpha=0.01)
plt.show()
# plt.savefig("sne_nst_b0_rand0_visualisation.pdf")
