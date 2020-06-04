import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.collections import LineCollection
from exp_config import *

variant = "aurora"
random = "0.0"
GEN_NUMBER = 6000

vae_loss = "fulllosstrue"
# _{vae_loss}

BASE_PATH = '/home/andwang1/airl/balltrajectorysd/results_exp1/second_run/'
EXP_PATH = f'results_balltrajectorysd_{variant}/gen6001_random{random}/'
os.chdir(BASE_PATH+EXP_PATH)
PID = os.listdir()[0] + "/"
os.chdir(BASE_PATH)
FILE_NAME = f'archive_{GEN_NUMBER}.dat'

FILE = BASE_PATH + EXP_PATH + PID + FILE_NAME

# descriptors = []
phenotypes = []

with open(FILE, "r") as f:
    for line in f.readlines():
        data = line.strip().split(" ")
        # descriptors.append(data[1,2])
        phenotypes.append([float(x) for x in data[-2:]])

fig = plt.figure()
spec = fig.add_gridspec(1, 1)
ax1 = fig.add_subplot(spec[0, 0], aspect='equal', adjustable='box')
max_dpf = max(phen[1] for phen in phenotypes)
plt.xlim([-max_dpf, max_dpf])
plt.ylim([-max_dpf, max_dpf])

# create lines, origin to point
lines = []
colours = []
for (angle, dpf) in phenotypes:
    x = np.cos(angle) * dpf
    y = np.sin(angle) * dpf
    lines.append([(0, 0), (x, y)])
    colours.append(dpf)

# Create a continuous norm to map from data points to colors
norm = plt.Normalize(0, max_dpf)

# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

# add all lines with the colours
lc = LineCollection(lines, cmap='PuBuGn', norm=norm)
colour_data = ax1.add_collection(lc)
fig.colorbar(colour_data, ax=ax1)

# Set the values used for colormapping
lc.set_array(np.array(colours))
lc.set_linewidth(1)
plt.title(f"Solution space in Polar Coordinates - Gen {GEN_NUMBER}")
plt.savefig(f"phenocircle_{variant}_{random}.pdf")
# plt.show()