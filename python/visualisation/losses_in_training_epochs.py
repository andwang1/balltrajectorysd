import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pk
import numpy as np
import pandas as pd

sns.set_style("dark")
with open("/home/andwang1/airl/balltrajectorysd/singularity/balltrajectorysd.sif/git/sferes2/losses_epoch.dat", "r") as f:
    data = f.readlines()

epochs_l2 = []
epochs_logvar = []
for line in data:
    l2 = [float(i) for i in line.split(",")[:-1:2]]
    logvar = [float(i) for i in line.split(",")[1:-1:2]]
    epochs_l2.append(l2)
    epochs_logvar.append(logvar)

l2 = np.array(epochs_l2)
logvar = np.array(epochs_logvar)

plt.plot(range(len(l2.mean(axis=1))), l2.mean(axis=1), label="L2")
plt.plot(range(len(logvar.mean(axis=1))), logvar.mean(axis=1), label="Decoder Logvar")
plt.legend()
plt.title("First Training Iteration - L2 and Logvar")
plt.xlabel("Epoch")
plt.ylabel("L2 / Log Variance")
plt.savefig("/home/andwang1/Pictures/final_report/exp2/first_train_losses.pdf")