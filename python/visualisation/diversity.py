import matplotlib.pyplot as plt
import numpy as np
from exp_config import *
import os


VAE = True
variant = "vae"
random = "0.0"
GEN_NUMBER = 6000

vae_loss = "fulllossfalse"

BASE_PATH = '/home/andwang1/airl/balltrajectorysd/results_exp1/second_run/'
EXP_PATH = f'results_balltrajectorysd_{variant}/gen6001_random{random}/'
if VAE:
    EXP_PATH = f'results_balltrajectorysd_{variant}/gen6001_random{random}_{vae_loss}/'
os.chdir(BASE_PATH+EXP_PATH)
PID = os.listdir()[0] + "/"
os.chdir(BASE_PATH)
FILE_NAME = f'diversity{GEN_NUMBER}.dat'

FILE = BASE_PATH + EXP_PATH + PID + FILE_NAME

with open(FILE, "r") as f:
    lines = f.readlines()
    max_diversity = int(lines[0].strip().split(":")[-1])
    achieved_diversity = round(float(lines[1].strip()),5)
    # bitmap prints in reverse order
    diversity_grid = lines[2].strip().split(",")[::-1]

rows = []
column = []
counter_x = 0
for i in diversity_grid:
    column.append(float(i))
    counter_x += 1
    if counter_x >= DISCRETISATION:
        counter_x = 0
        rows.append(column)
        column = []

# plot colours
fig = plt.figure(figsize=(15,15))
plt.ylim([DISCRETISATION, 0])
plt.xlim([0, DISCRETISATION])

# vmin/vmax sets limits
color = plt.pcolormesh(rows, vmin=0, vmax=1)

# plot grid
plt.grid(which="both")
plt.xticks(range(DISCRETISATION), np.arange(0, ROOM_W, ROOM_W / DISCRETISATION))
plt.yticks(range(DISCRETISATION), np.arange(0, ROOM_H, ROOM_H / DISCRETISATION))
fig.colorbar(color)
plt.title(f"Diversity - BinsTransversed / TotalBins - Gen {GEN_NUMBER}")
plt.xlabel("X")
plt.ylabel("Y")

ax1 = fig.add_subplot()
textbox = f"Score: {achieved_diversity} of {max_diversity}"
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
ax1.text(0.787, 1.03, textbox, transform=ax1.transAxes, fontsize=12,
    verticalalignment='top', bbox=props)

if VAE:
    plt.savefig(f"diversity_{variant}_{random}_{vae_loss}.pdf")
else:
    plt.savefig(f"diversity_{variant}_{random}.pdf")
# plt.show()