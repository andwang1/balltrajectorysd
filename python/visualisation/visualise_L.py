import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pk
import numpy as np
import pandas as pd

sns.set_style("dark")
colours = ["blue", "brown", "grey", "green", "purple", "red", "pink", "orange"]
# /media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/l2nosampletrain/results_balltrajectorysd_vae

PATHS = ["/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/l2/results_balltrajectorysd_vae/gen6001_random0_fulllossfalse_beta1_extension0_lossfunc2_sampletrue",
        "/media/andwang1/SAMSUNG/MSC_INDIV/results_box2d_bsd_exp1/l2beta0/results_balltrajectorysd_vae/gen6001_random0_fulllossfalse_beta0_extension0_lossfunc2_sampletrue",
         ]
NAMES = ["L2", "L2beta0"]
FILE_NAME = "stat_modifier.dat"


data = {}

for PATH, name in zip(PATHS, NAMES):
    os.chdir(PATH)
    PIDs = [pid for pid in os.listdir() if os.path.isdir(pid)]

    gens = []
    l = []

    for pid in PIDs:
        os.chdir(os.path.join(PATH, pid))
        print(os.getcwd())
        with open(FILE_NAME, "r") as f:
            lines = f.readlines()
        for line in lines:
            elements = line.strip().split(" ")
            gens.append(int(elements[0]))
            l.append(float(elements[1]))

    data[name] = {"gens": gens, "l": l}


f = plt.figure(figsize=(10, 5))
spec = f.add_gridspec(1, 1)
ax1 = f.add_subplot(spec[0, 0])

colour_count = 0
for name in NAMES:
    sns.lineplot(data[name]["gens"], data[name]["l"], estimator=np.median, ci=None, ax=ax1, label=name, color=colours[colour_count])
    data_stats = pd.DataFrame({"x": data[name]["gens"], "y": data[name]["l"]}).groupby("x").describe()
    quart25 = data_stats[("y", '25%')]
    quart75 = data_stats[("y", '75%')]
    ax1.fill_between(list(range(len(lines))), quart25, quart75, alpha=0.3, color=colours[colour_count])
    colour_count += 1

plt.title("Value of Distance Threshold Parameter L")
ax1.set_ylabel("L")
ax1.set_xlabel("Generation")
# os.chdir(PATH)
# plt.savefig("extend_dist.png")
# plt.savefig("/home/andwang1/Pictures/final_report/exp2/L_progression.pdf")
# plt.savefig("dist_extension.png")
plt.show()